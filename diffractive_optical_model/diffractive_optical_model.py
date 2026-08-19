import math

import torch
from loguru import logger
from pytorch_lightning import LightningModule
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

from diffractive_optical_model.diffraction_block.diffraction_block import DiffractionBlock


_OPTIMIZERS = {
    'ADAM': torch.optim.Adam,
    'ADAMW': torch.optim.AdamW,
    'SGD': torch.optim.SGD,
}


class DOM(LightningModule):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.params = params
        self.training_params = params['dom_training']
        self.data_range = float(self.training_params.get('data_range', 1.0))
        if not math.isfinite(self.data_range) or self.data_range <= 0:
            raise ValueError("dom_training.data_range must be finite and strictly positive.")
        self.select_objective()
        self.create_layers()
        self.learning_rate = self.training_params['learning_rate']
        self.save_hyperparameters()

    def configure_optimizers(self):
        name = str(self.training_params.get('optimizer', 'ADAM')).upper()
        if name not in _OPTIMIZERS:
            raise ValueError(
                "Optimizer {} is not supported; use one of {}".format(name, list(_OPTIMIZERS))
            )
        logger.debug("DOM | setting optimizer to {}".format(name))
        trainable_parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise ValueError(
                "DOM has no trainable parameters. Set at least one modulator gradients mode "
                "to 'phase_only', 'amplitude_only', or 'complex'."
            )
        return _OPTIMIZERS[name](trainable_parameters, lr=self.learning_rate)

    def select_objective(self):
        objective_function = str(self.training_params['objective_function']).lower()
        self.objective_name = objective_function
        if objective_function == "mse":
            self.similarity_metric = False
            self.objective_function = torch.nn.functional.mse_loss
        elif objective_function == "psnr":
            self.similarity_metric = True
            self.objective_function = psnr
        elif objective_function == "ssim":
            self.similarity_metric = True
            self.objective_function = ssim
        else:
            raise ValueError("Objective function : {} not supported".format(objective_function))
        logger.debug("DOM | setting objective function to {}".format(objective_function))

    def create_layers(self):
        self.layers = torch.nn.ModuleList()
        bits = self.params.get('bits', 64)
        for block in self.params['diffraction_blocks']:
            block_params = dict(self.params['diffraction_blocks'][block])
            block_params.setdefault('bits', bits)
            self.layers.append(DiffractionBlock(block_params))

    def run_dom_metrics(self, dom_outputs, targets):
        images = dom_outputs['images'].detach()
        target_images = self._target_intensity(targets).detach()
        mse_vals = torch.nn.functional.mse_loss(images, target_images)
        psnr_vals = psnr(images, target_images, data_range=self.data_range)
        ssim_vals = ssim(
            images, target_images, data_range=self.data_range
        ).detach()
        return {'mse': mse_vals.cpu(), 'psnr': psnr_vals.cpu(), 'ssim': ssim_vals.cpu()}

    def objective(self, output, target):
        target = self._target_intensity(target)
        if output.is_complex():
            output = output.abs() ** 2
        if self.objective_name == "mse":
            return torch.nn.functional.mse_loss(input=output, target=target)
        if self.objective_name == "psnr":
            error = torch.nn.functional.mse_loss(input=output, target=target)
            epsilon = torch.finfo(error.dtype).eps
            stable_psnr = 10 * torch.log10(
                output.new_tensor(self.data_range ** 2) / error.clamp_min(epsilon)
            )
            return -stable_psnr
        if self.objective_name == "ssim":
            similarity = ssim(
                preds=output, target=target, data_range=self.data_range
            )
            return 1 - similarity
        raise RuntimeError("DOM objective was not initialized correctly.")

    @staticmethod
    def _target_intensity(target):
        return target.abs() ** 2 if target.is_complex() else target

    def calculate_auxiliary_outputs(self, output_wavefronts) -> dict:
        amplitudes = output_wavefronts.abs()
        amax = amplitudes.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
        normalized_amplitudes = amplitudes / amax
        images = amplitudes ** 2
        imax = images.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
        normalized_images = images / imax
        return {
            'output_wavefronts': output_wavefronts,
            'amplitudes': amplitudes,
            'normalized_amplitudes': normalized_amplitudes,
            'images': images,
            'normalized_images': normalized_images,
        }

    def forward(self, u: torch.Tensor):
        for layer in self.layers:
            u = layer(u)
        return u

    def shared_step(self, batch, batch_idx):
        samples, targets = batch
        output_wavefronts = self.forward(samples)
        outputs = self.calculate_auxiliary_outputs(output_wavefronts)
        return outputs, targets

    def training_step(self, batch, batch_idx):
        outputs, targets = self.shared_step(batch, batch_idx)
        loss = self.objective(outputs['images'], targets)
        self.log("train_loss", loss, prog_bar=True)
        return {'loss': loss, 'outputs': outputs, 'target': targets.detach()}

    def validation_step(self, batch, batch_idx):
        outputs, targets = self.shared_step(batch, batch_idx)
        loss = self.objective(outputs['images'], targets)
        self.log("val_loss", loss, prog_bar=True)
        return {'loss': loss, 'output': outputs, 'target': targets.detach()}
