import torch
from .strategy import FFTStrategy

class MTPFFTStrategy(FFTStrategy):
    def __init__(self, input_plane, output_plane):
        self.input_plane = input_plane
        self.output_plane = output_plane

    def create_dft_matrices(self):
        dx_input = self.input_plane.delta_x
        dy_input = self.input_plane.delta_y

        dx_output = self.output_plane.delta_x
        dy_output = self.output_plane.delta_y

        if dx_input.real < dx_output.real:
            fx = self.output_plane.fx_padded
        else:
            fx = self.input_plane.fx_padded

        if dy_input.real < dy_output.real:
            fy = self.output_plane.fy_padded
        else:
            fy = self.input_plane.fy_padded

        self.dft_matrix_x = torch.exp(-2j * torch.pi * torch.outer(fx, self.input_plane.x_padded)).unsqueeze(0)
        self.dft_matrix_y = torch.exp(-2j * torch.pi * torch.outer(fy, self.input_plane.y_padded)).unsqueeze(0)

    def create_idft_matrices(self):
        # I might need to double these here
        M_output = self.output_plane.Nx
        N_output = self.output_plane.Ny

        dx_input = self.input_plane.delta_x
        dy_input = self.input_plane.delta_y
        dx_output = self.output_plane.delta_x
        dy_output = self.output_plane.delta_y

        if dx_input.real < dx_output.real:
            fx = self.output_plane.fx_padded
        else:
            fx = self.input_plane.fx_padded

        if dy_input.real < dy_output.real:
            fy = self.output_plane.fy_padded
        else:
            fy = self.input_plane.fy_padded

        self.idft_matrix_x = torch.exp(2j * torch.pi * torch.outer(self.output_plane.x_padded, fx)).unsqueeze(0) / M_output
        self.idft_matrix_y = torch.exp(2j * torch.pi * torch.outer(self.output_plane.y_padded, fy)).unsqueeze(0) / N_output

    def fft(self, g):
       # Perform the DFT along x-axis
       g_dft_x = self.dft_matrix_x[0] @ g

       # Perform the DFT along y-axis
       g_dft_xy = self.dft_matrix_y[0] @ g_dft_x.permute(0, 2, 1)
       g_dft_xy = g_dft_xy.permute(0, 2, 1)
       return g_dft_xy

    def ifft(self, G):
       # Perform the DIFT using dot product along y-axis (columns)
       g_reconstructed_y = self.idft_matrix_y[0] @ G.permute(0, 2, 1)

       # Perform the DIFT using dot product along x-axis (rows)
       g_reconstructed = self.idft_matrix_x[0] @ g_reconstructed_y.permute(0, 2, 1)

       return g_reconstructed.unsqueeze(1)

