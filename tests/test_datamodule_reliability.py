import pytest
import torch

from diffractive_optical_model.datamodule import custom_transforms


def test_normalize_handles_uint8_and_constant_samples():
    sample = torch.tensor([[[0, 51, 255]]], dtype=torch.uint8)
    normalized = custom_transforms.Normalize({})(sample)
    thresholded = custom_transforms.Threshold(0.2)(normalized)

    assert normalized.dtype.is_floating_point
    assert torch.allclose(normalized, torch.tensor([[[0.0, 0.2, 1.0]]]))
    assert torch.equal(thresholded, torch.tensor([[[0.0, 0.0, 1.0]]]))
    assert torch.equal(
        custom_transforms.Normalize({})(torch.full((1, 2, 2), 7, dtype=torch.uint8)),
        torch.zeros(1, 2, 2),
    )


@pytest.mark.parametrize(
    ("bits", "expected_dtype"),
    [(64, torch.complex64), (128, torch.complex128)],
)
def test_wavefront_transform_honors_bits_and_zero_phase(bits, expected_dtype):
    transform = custom_transforms.WavefrontTransform(
        {"phase_initialization_strategy": 0, "bits": bits}
    )
    sample = torch.full((1, 2, 2), 0.5)
    wavefront = transform(sample)

    assert wavefront.dtype == expected_dtype
    assert torch.allclose(wavefront.real, sample.to(wavefront.real.dtype))
    assert torch.count_nonzero(wavefront.imag) == 0


def test_seeded_split_reserves_official_test_and_handles_zero_workers(monkeypatch):
    pytest.importorskip("torchvision")
    from diffractive_optical_model.datamodule import datamodule

    class FakeMNIST:
        def __init__(self, _path, train, download=False):
            self.train = train
            count = 10 if train else 4
            self.data = torch.arange(count * 4 * 4, dtype=torch.uint8).reshape(count, 4, 4)
            self.targets = torch.arange(count)

        def __len__(self):
            return len(self.data)

    monkeypatch.setattr(datamodule, "MNIST", FakeMNIST)
    params = {
        "Nxp": 5,
        "Nyp": 7,
        "n_cpus": 0,
        "paths": {"path_data": "data", "path_root": "."},
        "batch_size": 2,
        "resize_row": 4,
        "resize_col": 6,
        "valid_rate": 0.2,
        "seed": [True, 17],
        "bits": 64,
        "wavefront_transform": {"phase_initialization_strategy": 0},
    }

    first = datamodule.Wavefront_MNIST_DataModule(params)
    second = datamodule.Wavefront_MNIST_DataModule(params)
    first.setup()
    second.setup("fit")

    assert len(first.mnist_train) == 8
    assert len(first.mnist_val) == 2
    assert first.mnist_val.indices == second.mnist_val.indices
    assert first.mnist_val.dataset.samples.shape[0] == 10
    assert first.mnist_test.samples.shape[0] == 4
    assert first.train_dataloader().persistent_workers is False

    sample, _ = first.mnist_train.dataset[0]
    assert sample.shape == (1, 5, 7)
