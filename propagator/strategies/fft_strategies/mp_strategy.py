import torch
from strategy import FFTStrategy

class MPFFTStrategy(FFTStrategy):
    def __init__(self, input_plane, output_plane):
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.pick_fx_fy()
        self.create_dft_matrices()
        self.create_idft_matrices()

    def pick_fx_fy(self):
        dx_input = self.input_plane.delta_x
        dy_input = self.input_plane.delta_y

        dx_output = self.output_plane.delta_x
        dy_output = self.output_plane.delta_y

        # Which sample spacing is limiting in terms of the possible frequencies
        if dx_input.real < dx_output.real:
            self.fx = self.output_plane.fx
        else:
            self.fx = self.input_plane.fx

        if dy_input.real < dy_output.real:
            self.fy = self.output_plane.fy
        else:
            self.fy = self.input_plane.fy

    def create_dft_matrices(self):
        self.dft_matrix_x = torch.exp(-2j * torch.pi * torch.outer(self.fx, self.input_plane.x)).unsqueeze(0)
        self.dft_matrix_y = torch.exp(-2j * torch.pi * torch.outer(self.fy, self.input_plane.y)).unsqueeze(0)

    def create_idft_matrices(self):
        # I might need to double these here
        M_output = self.output_plane.Nx
        N_output = self.output_plane.Ny

        self.idft_matrix_x = torch.exp(2j * torch.pi * torch.outer(self.output_plane.x, self.fx)).unsqueeze(0) / M_output
        self.idft_matrix_y = torch.exp(2j * torch.pi * torch.outer(self.output_plane.y, self.fy)).unsqueeze(0) / N_output

    def fft(self, g):
        g_dft = self.dft_matrix_x[0] @ g.transpose(0, 1)
        return g_dft.T

    def ifft(self, G):
        g_reconstructed_x = self.idft_matrix_x[0] @ G.transpose(0, 1)
        return g_reconstructed_x.T

    def fft2(self, g):
       # Perform the DFT along x-axis
       g_dft_x = self.dft_matrix_x @ g

       # Perform the DFT along y-axis
       g_dft_xy = self.dft_matrix_y @ g_dft_x.permute(0, 2, 1)
       g_dft_xy = g_dft_xy.permute(0, 2, 1)
       return g_dft_xy

    def ifft2(self, G):
       # Perform the DIFT using dot product along y-axis (columns)
       g_reconstructed_y = self.idft_matrix_y[0] @ G.permute(0, 2, 1)

       # Perform the DIFT using dot product along x-axis (rows)
       g_reconstructed = self.idft_matrix_x[0] @ g_reconstructed_y.permute(0, 2, 1)

       return g_reconstructed.unsqueeze(1)

