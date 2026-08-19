import torch
from diffractive_optical_model.propagator.strategies.fft_strategies.strategy import FFTStrategy


class MPFFTStrategy(FFTStrategy):
    """Deprecated matrix-product DFT retained for same-grid regression tests.

    Its arbitrary mismatched-grid normalization is not physically valid and
    the public factory refuses to select it. Use ``CZTStrategy`` instead.
    """

    def __init__(self, input_plane, output_plane, kwargs=None):
        super().__init__()
        if kwargs is None:
            kwargs = {}
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.padded = kwargs.get('padded', False)
        self.pick_fx_fy()
        self.pick_x_y()
        self.create_dft_matrices()
        self.create_idft_matrices()

    def __repr__(self):
        return (
            f"MPFFTStrategy(input_plane={self.input_plane}, "
            f"output_plane={self.output_plane}, padded={self.padded})"
        )

    def pick_fx_fy(self):
        dx_input = self.input_plane.delta_x
        dy_input = self.input_plane.delta_y
        dx_output = self.output_plane.delta_x
        dy_output = self.output_plane.delta_y

        # Coarser pitch limits the representable frequencies.
        if dx_input <= dx_output:
            src = self.output_plane
        else:
            src = self.input_plane
        if self.padded:
            self.fx = src.fx_padded
            self.delta_fx = src.delta_fx_padded
        else:
            self.fx = src.fx
            self.delta_fx = src.delta_fx

        if dy_input <= dy_output:
            src_y = self.output_plane
        else:
            src_y = self.input_plane
        if self.padded:
            self.fy = src_y.fy_padded
            self.delta_fy = src_y.delta_fy_padded
        else:
            self.fy = src_y.fy
            self.delta_fy = src_y.delta_fy

        self.fxx, self.fyy = torch.meshgrid(self.fx, self.fy, indexing='ij')

    def pick_x_y(self):
        if self.padded:
            self.x_input = self.input_plane.x_padded
            self.y_input = self.input_plane.y_padded
            self.xx_input = self.input_plane.xx_padded
            self.yy_input = self.input_plane.yy_padded
            self.x_output = self.output_plane.x_padded
            self.y_output = self.output_plane.y_padded
            self.xx_output = self.output_plane.xx_padded
            self.yy_output = self.output_plane.yy_padded
        else:
            self.x_input = self.input_plane.x
            self.y_input = self.input_plane.y
            self.xx_input = self.input_plane.xx
            self.yy_input = self.input_plane.yy
            self.x_output = self.output_plane.x
            self.y_output = self.output_plane.y
            self.xx_output = self.output_plane.xx
            self.yy_output = self.output_plane.yy

    def create_dft_matrices(self):
        x_shift = torch.fft.ifftshift(self.x_input)
        y_shift = torch.fft.ifftshift(self.y_input)
        dft_matrix_x = torch.exp(-2j * torch.pi * torch.outer(self.fx, x_shift)).unsqueeze(0)
        dft_matrix_y = torch.exp(-2j * torch.pi * torch.outer(self.fy, y_shift)).unsqueeze(0)
        self.register_buffer('dft_matrix_x', dft_matrix_x)
        self.register_buffer('dft_matrix_y', dft_matrix_y)

    def create_idft_matrices(self):
        n_fx = self.fx.numel()
        n_fy = self.fy.numel()
        x_shift = torch.fft.ifftshift(self.x_output)
        y_shift = torch.fft.ifftshift(self.y_output)
        idft_matrix_x = torch.exp(2j * torch.pi * torch.outer(x_shift, self.fx)).unsqueeze(0) / n_fx
        idft_matrix_y = torch.exp(2j * torch.pi * torch.outer(y_shift, self.fy)).unsqueeze(0) / n_fy
        self.register_buffer('idft_matrix_x', idft_matrix_x)
        self.register_buffer('idft_matrix_y', idft_matrix_y)

    def fft(self, g):
        orig = g.dtype
        g = torch.fft.ifftshift(g, dim=-1).to(self.dft_matrix_x.dtype)
        g_dft = self.dft_matrix_x[0] @ g.transpose(-2, -1)
        return g_dft.transpose(-2, -1).to(orig)

    def ifft(self, G):
        orig = G.dtype
        G = G.to(self.idft_matrix_x.dtype)
        g = self.idft_matrix_x[0] @ G.transpose(-2, -1)
        g = g.transpose(-2, -1)
        return torch.fft.fftshift(g, dim=-1).to(orig)

    def fft2(self, g):
        orig = g.dtype
        g = torch.fft.ifftshift(g, dim=(-2, -1)).to(self.dft_matrix_x.dtype)
        out = self.dft_matrix_x @ g @ self.dft_matrix_y.transpose(-2, -1)
        return out.to(orig)

    def ifft2(self, G):
        orig = G.dtype
        G = G.to(self.idft_matrix_x.dtype)
        g = self.idft_matrix_x @ G @ self.idft_matrix_y.transpose(-2, -1)
        return torch.fft.fftshift(g, dim=(-2, -1)).to(orig)
