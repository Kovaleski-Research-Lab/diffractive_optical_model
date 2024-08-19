#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import torch
import numpy as np
from loguru import logger
import torchvision
import pytorch_lightning as pl
from IPython import embed

#from . import plane
import plane
from dft import DFT, DIFT


#--------------------------------
# Class: PropagatorFactory
#--------------------------------
class PropagatorFactory():
    def __call__(self, input_plane:plane.Plane, output_plane:plane.Plane, params:dict):
        return self.select_propagator(input_plane, output_plane, params)

    def select_propagator(self, input_plane, output_plane, params):
        logger.debug("Selecting propagator")
        ###
        # Propagator types:
        #   DNI: Direct numerical integration
        #   ASM: Angular spectrum method
        #   RSC: Rayleigh-Sommerfeld convolution
        #   Scaled: Scaled angular spectrum method
        #   Rotated: Rotated angular spectrum method #TODO

        # Propagator references:
        #   ASM: 10.1364/JOSAA.401908 
        #   RSC: 10.1364/JOSAA.401908
        #   Shift: 10.1364/OE.18.018453 
        #   Scaled: 10.1364/JOSAA.31.001832, 10.1364/JOSAA.29.002415
        #   Rotated: 10.1364/JOSAA.20.001755

        # This function compares the input and output planes to determine 
        # which propagator type to use and in which order to apply the propagators.

        # The shift is a special case of the ASM and RSC propagation methods where
        # the transfer function is shifted to account for the shift between the input
        # and output planes. This is done using the shift theorem.

        # The scaled propagation is a special case of the ASM and RSC propagation
        # methods where the chirp z transform is used to perform the IFFT. This
        # scales the output appropriately.
        ###

        # Check: Are the input and output planes tilted? If so, create a rotation
        # matrix to rotate both planes to the same orientation.
        if (input_plane.normal != output_plane.normal).any():
            logger.debug("Input and output planes are tilted. Creating rotation matrix")
            # This is not used yet
            #rot = self.create_rotation_matrix(input_plane.normal, output_plane.normal)
            logger.error("Rotation of input and output planes is not implemented yet.")
            raise NotImplementedError("Rotation of input and output planes is not implemented yet.")

        
        # If the wavelength is not a tensor, convert it to a tensor.
        if not torch.is_tensor(params['wavelength']):
            wavelength = torch.tensor(params['wavelength'])
        else:
            wavelength = params['wavelength']

        # Create the DFT and DIFT functions
        self.dft = DFT(input_plane, output_plane)
        self.dift = DIFT(input_plane, output_plane)

        # Pick the propagator
        # Used the specifed. Else, check the distance and use the appropriate propagator
        # to avoid aliasing.
        if params['prop_type'] == 'dni':
            logger.debug("Using DNI propagation")
            prop_function = None
            prop_type = 'dni'
        elif params['prop_type'] == 'asm':
            logger.debug("Using ASM propagation")
            prop_function = self.init_asm_transfer_function(input_plane, output_plane, wavelength)
            prop_type = 'asm'
        elif params['prop_type'] == 'rsc':
            logger.debug("Using RSC propagation")
            prop_function = self.init_rsc_transfer_function(input_plane, output_plane, wavelength)
            prop_type = 'rsc'
        elif params['prop_type'] == None and self.check_asm_distance(input_plane, output_plane, params):
            logger.debug("Using ASM propagator")
            prop_function = self.init_asm_transfer_function(input_plane, output_plane, wavelength)
            prop_type = 'asm'
        elif params['prop_type'] == None:
            logger.debug("Using RSC propagator")
            prop_function = self.init_rsc_transfer_function(input_plane, output_plane, wavelength)
            prop_type = 'rsc'
        else:
            logger.error("Invalid propagation type: {}".format(params['prop_type']))
            raise ValueError("Invalid propagation type: {}".format(params['prop_type']))

        

        # Create the propagator
        propagator = Propagator(input_plane, output_plane, prop_function, prop_type, self.dft, self.dift)
        return propagator


    def create_rotation_matrix(self, input_normal, output_normal):
        logger.debug("Creating rotation matrix")
        # This function creates a rotation matrix to rotate the input plane to
        # the orientation of the output plane. The rotation matrix is returned.

        # Normalize the input and output normals
        input_normal = input_normal / torch.norm(input_normal)
        output_normal = output_normal / torch.norm(output_normal)

        if (input_normal.isnan()).any() or (output_normal.isnan()).any():
            logger.debug("Input or output plane normal is nan.")
            return torch.eye(3)

        if (input_normal == output_normal).all():
            logger.debug("Input and output plane normals are the same.")
            return torch.eye(3)

        if (input_normal == 0).all() or (output_normal == 0).all():
            logger.debug("Input or output plane normal is zero.")
            return torch.eye(3)

        if self.are_antiparallel(input_normal, output_normal):
            logger.debug("Input and output plane normals are antiparallel.")
            return torch.eye(3)

        rot_axis = torch.cross(input_normal, output_normal)
        rot_axis = rot_axis / torch.norm(rot_axis)
        rot_angle = torch.acos(torch.dot(input_normal, output_normal))
        rot_matrix = self.create_rotation_matrix_from_axis_angle(rot_axis, rot_angle)
        logger.debug("Rotation matrix: {}".format(rot_matrix))

        return rot_matrix

    def are_antiparallel(self, input_normal, output_normal):
        logger.debug("Checking if input and output plane normals are antiparallel")
        # This function checks if the input and output plane normals are antiparallel.
        # The result is returned.
        return torch.allclose(input_normal, -output_normal)

    def create_rotation_matrix_from_axis_angle(self, axis, angle):
        logger.debug("Creating rotation matrix from axis and angle")
        # This function creates a rotation matrix from an axis and angle.
        # The rotation matrix is returned.
        axis = axis / torch.norm(axis)
        a = torch.cos(angle / 2.0)
        b, c, d = -axis * torch.sin(angle / 2.0)
        rot_matrix = torch.tensor([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                                   [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                                   [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
        return rot_matrix

    def check_asm_distance(self, input_plane, output_plane, params):
        logger.debug("Checking ASM propagation criteria")
        #10.1364/JOSAA.401908 equation 32
        #Checks distance criteria for sampling considerations
        wavelength = torch.tensor(params['wavelength'])
        delta_x = input_plane.delta_x
        delta_y = input_plane.delta_y
        Nx = input_plane.Nx
        Ny = input_plane.Ny

        shift_x = output_plane.center[0] - input_plane.center[0]
        shift_y = output_plane.center[1] - input_plane.center[1]
        distance = output_plane.center[-1] - input_plane.center[-1]
        distance = torch.abs(distance)

        logger.debug("Axial distance between input and output planes: {}".format(distance))
        logger.debug("Shift between input and output planes: [{}, {}]".format(shift_x, shift_y))

        distance_criteria_y = 2 * delta_y * ( Ny * delta_y - shift_y) / wavelength
        distance_criteria_y *= torch.sqrt(1 - (wavelength / (2 * Ny))**2)
        distance_criteria_y = torch.abs(distance_criteria_y)
       
        distance_criteria_x = 2 * delta_x * ( Nx * delta_x - shift_x) / wavelength
        distance_criteria_x *= torch.sqrt(1 - (wavelength / (2 * Nx))**2)
        distance_criteria_x = torch.abs(distance_criteria_x)
        
        strict_distance = torch.min(distance_criteria_y, distance_criteria_x) 
        logger.debug("Maximum axial distance for asm : {}".format(strict_distance))
    
        return(torch.le(distance, strict_distance))

    def init_asm_transfer_function(self, input_plane, output_plane, wavelength): 
        logger.debug("Initializing ASM transfer function")

        input_dx = input_plane.delta_x.real
        input_dy = input_plane.delta_y.real
        output_dx = output_plane.delta_x.real
        output_dy = output_plane.delta_y.real

        # Check which is the limiting plane
        # Use the plane with the larger dx and dy
        if input_dx > output_dx:
            fxx = input_plane.fxx_padded
        else:
            fxx = output_plane.fxx_padded

        if input_dy > output_dy:
            fyy = input_plane.fyy_padded
        else:
            fyy = output_plane.fyy_padded

        #Mask out non-propagating waves
        mask = torch.sqrt(fxx**2 + fyy**2).real < (1/wavelength)
        fxx = mask * fxx
        fyy = mask * fyy
        
        #10.1364/JOSAA.401908 equation 28
        #Also Goodman eq 3-78
        fz = torch.sqrt(1 - (wavelength*fxx)**2 - (wavelength*fyy)**2).real
        fz *= ((torch.pi * 2)/wavelength)

        # Get the axial distance between the input and output planes.
        # NOTE: When we move to rotation propatation, this will need to be updated
        # to get the distance between the planes along the propagation axis.
        #distance = torch.norm(output_plane.center - input_plane.center)
        distance = output_plane.center[-1] - input_plane.center[-1]

        # Initialize the transfer function
        H = torch.exp(1j * distance * fz)

        # Normalize the transfer function
        mag = H.abs()
        ang = H.angle()
        mag = mag / torch.max(mag)
        H = mag * torch.exp(1j*ang)

        # Update the transfer function to account for the shift between the input and output planes.
        # Get the x-y shift between the input and output planes.
        shift = output_plane.center - input_plane.center
        x_shift = shift[0]
        y_shift = shift[1]

        # Shift the transfer function using shift theorem
        H = H * torch.exp(1j * 2 * torch.pi * (fxx * -x_shift + fyy * -y_shift))
        H = torch.fft.fftshift(H)

        H.requires_grad = False

        return H

    def init_rsc_transfer_function(self, input_plane, output_plane, wavelength):
        logger.debug("Initializing RSC transfer function")

        # Get the spatial dimensions of the input plane
        xx, yy = input_plane.xx_padded, input_plane.yy_padded

        fxx, fyy = input_plane.fxx_padded, input_plane.fyy_padded

        #Mask out non-propagating waves
        mask = torch.sqrt(fxx**2 + fyy**2).real < (1/wavelength)
        fxx = mask * fxx
        fyy = mask * fyy

        # Get the axial distance between the input and output planes.
        # NOTE: When we move to rotation propatation, this will need to be updated
        # to get the distance between the planes along the propagation axis.
        #distance = torch.norm(output_plane.center - input_plane.center)
        distance = output_plane.center[-1] - input_plane.center[-1]

        # Get the x-y shift between the input and output planes.
        shift = output_plane.center - input_plane.center
        x_shift = shift[0]
        y_shift = shift[1]

        #10.1364/JOSAA.401908 equation 29
        #Also Goodman eq 3-79
        r = torch.sqrt((xx - x_shift)**2 + (yy - y_shift)**2 + distance**2).real
        k = (2 * torch.pi / wavelength)
        z = distance

        # Initialize the impulse response
        h_rsc = torch.exp(torch.sign(distance) * 1j*k*r) / r
        h_rsc *= ((1/r) - (1j*k))
        h_rsc *= (1/(2*torch.pi)) * (z/r)

        # Get the transfer function
        H = self.dft(h_rsc)

        # Normalize the transfer function
        mag = H.abs()
        ang = H.angle()
        mag = mag / torch.max(mag)
        #mag = torch.ones(H.size())
        H = mag * torch.exp(1j*ang)
        H = torch.fft.fftshift(H)
        
        H.requrires_grad = False
        return H

class Propagator(pl.LightningModule):
    def __init__(self, input_plane, output_plane, transfer_function, prop_type, dft=torch.fft.fftn, dift=torch.fft.ifftn):
        super().__init__()
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.transfer_function = transfer_function
        self.prop_type = prop_type
        self.register_buffer('H', self.transfer_function)
        self.cc_output = torchvision.transforms.CenterCrop((int(output_plane.Nx), int(output_plane.Ny)))
        self.cc_input = torchvision.transforms.CenterCrop((int(input_plane.Nx), int(input_plane.Ny)))
        padx = torch.div(input_plane.Nx, 2, rounding_mode='trunc')
        pady = torch.div(input_plane.Ny, 2, rounding_mode='trunc')
        self.padding = (pady,pady,padx,padx)    
        self.dft = dft
        self.dift = dift

    def forward(self, input_wavefront):
        # Pad the wavefront
        input_wavefront = torch.nn.functional.pad(input_wavefront,self.padding,mode="constant") # type: ignore
        return self.propagate(input_wavefront)

    def propagate(self, input_wavefront):
        ###
        # Propagates the wavefront from the input plane to the output plane.
        ###
        if self.prop_type == 'asm':
            output_wavefront = self.asm_propagate(input_wavefront)
        elif self.prop_type == 'rsc':
            output_wavefront = self.rsc_propagate(input_wavefront)
        elif self.prop_type == 'dni':
            output_wavefront = self.dni_propagate(input_wavefront)
        else:
            logger.error("Invalid propagation type: {}".format(self.prop_type))
            raise ValueError("Invalid propagation type: {}".format(self.prop_type))

        return output_wavefront

    def asm_propagate(self, input_wavefront):
        #logger.debug("Propagating using ASM")
        ###
        # Propagates the wavefront using the angular spectrum method.
        ###
        A = self.dft(input_wavefront)
        A = torch.fft.fftshift(A)
        from IPython import embed; embed()
        U = A * self.H
        U = torch.fft.ifftshift(U, dim=(-1,-2))
        U = self.cc_output(U)
        U = self.dift(U)
        return U

    def rsc_propagate(self, input_wavefront):
        #logger.debug("Propagating using RSC")
        ###
        # Propagates the wavefront using the rayleigh-sommerfeld convolution.
        ###
        #A = torch.fft.fft2(input_wavefront)
        A = self.dft(input_wavefront)
        A = torch.fft.fftshift(A)
        U = A * self.H 
        U = torch.fft.ifftshift(U, dim=(-1,-2))
        U = self.dift(U)
        U = self.cc_output(U)
        return U

    def dni_propagate(self, input_wavefront):
        #logger.debug("Propagating using DNI")
        ###
        # Propagates the wavefront using direct numerical integration.
        # I think this assumes axis aligned planes propagating in the z direction
        ###
        z_distance = self.input_plane.center[-1] - self.output_plane.center[-1]
        k = torch.pi * 2 / self.wavelength
        output_field = input_wavefront.new_empty(input_wavefront.size(), dtype=torch.complex64)

        from tqdm import tqdm
        for i,x in enumerate(tqdm(self.input_plane.x_padded)):
            for j,y in enumerate(self.input_plane.y_padded):
                r = torch.sqrt((self.output_plane.xx_padded-x)**2 + (self.output_plane.yy_padded-y)**2 + z_distance**2)
                chirp = torch.exp(1j * k * r)
                scalar1 = z_distance / r
                scalar2 = (( 1 / r) - 1j*k)
                combined = input_wavefront * chirp * scalar1 * scalar2
                output_field[:,:,i,j] = combined.sum()

        return output_field

    def print_info(self):

        logger.debug("Propagator info:")
        logger.debug("Input plane: {}".format(self.input_plane))
        logger.debug("Output plane: {}".format(self.output_plane))
        logger.debug("Transfer function: {}".format(self.transfer_function))
        logger.debug("Propagation type: {}".format(self.prop_type))

#--------------------------------
# Initialize: Test code
#--------------------------------

if __name__ == "__main__":

    input_plane_params = {
        'name': 'input_plane',
        'size': torch.tensor([8.96e-3, 8.96e-3]),
        'Nx': 1000,
        'Ny': 1000,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0,0,0])
    }

    output_plane_params0 = {
        'name': 'output_plane',
        'size': torch.tensor([2.96e-3, 2.96e-3]),
        'Nx': 2000,
        'Ny': 2000,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0,0,11e-2])
    }

    output_plane_params1 = {
        'name': 'output_plane',
        'size': torch.tensor([8.96e-3, 8.96e-3]),
        'Nx': 500,
        'Ny': 500,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0,0,20e-2])
    }

    input_plane = plane.Plane(input_plane_params)
    output_plane0 = plane.Plane(output_plane_params0)
    output_plane1 = plane.Plane(output_plane_params0)

    propagator_params = {
        'prop_type': 'asm',
        'wavelength': torch.tensor(1.55e-6),
    }

    propagator0 = PropagatorFactory()(input_plane, output_plane0, propagator_params)
    propagator_params['prop_type'] = 'rsc'
    propagator1 = PropagatorFactory()(input_plane, output_plane1, propagator_params)

    # Example wavefront to propagate
    # This is a plane wave through a 1mm aperture
    x = torch.linspace(-input_plane.Lx/2, input_plane.Lx/2, input_plane.Nx)
    y = torch.linspace(-input_plane.Ly/2, input_plane.Ly/2, input_plane.Ny)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    wavefront = torch.ones_like(xx)
    wavefront[(xx**2 + yy**2) > (0.6e-3)**2] = 0
    wavefront = wavefront.view(1,1,input_plane.Nx,input_plane.Ny)
    wavefront = wavefront.type(torch.complex128)

    # Propagate the wavefront
    output_wavefront0 = propagator0(wavefront).squeeze()
    output_wavefront1 = propagator1(wavefront).squeeze()
    #output_wavefront2 = propagator2(wavefront)

    # Plot the input and output wavefronts
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib import ticker

    scale = 1e3
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*scale))
    
    fig, axes = plt.subplots(1,3, figsize=(20,5))
    im0 = axes[0].imshow(wavefront.abs().numpy().squeeze(), extent=[input_plane.x.real.min(), input_plane.x.real.max(), input_plane.y.real.min(), input_plane.y.real.max()])
    axes[0].xaxis.set_major_formatter(ticks)
    axes[0].yaxis.set_major_formatter(ticks)
    axes[0].set_title("Input wavefront")
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax)

    im1 = axes[1].imshow(output_wavefront0.abs().numpy(), extent=[output_plane0.x.real.min(), output_plane0.x.real.max(), output_plane0.y.real.min(), output_plane0.y.real.max()])
    axes[1].xaxis.set_major_formatter(ticks)
    axes[1].yaxis.set_major_formatter(ticks)
    axes[1].set_title("Output wavefront (ASM)")
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)

    im2 = axes[2].imshow(output_wavefront1.abs().numpy(), extent=[output_plane1.x.real.min(), output_plane1.x.real.max(), output_plane1.y.real.min(), output_plane1.y.real.max()])
    axes[2].xaxis.set_major_formatter(ticks)
    axes[2].yaxis.set_major_formatter(ticks)
    axes[2].set_title("Output wavefront (RSC)")
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)

    #im3 = axes[3].imshow(difference)
    #axes[3].set_title("Difference")
    #divider = make_axes_locatable(axes[3])
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #plt.colorbar(im3, cax=cax)

    #axes[3].set_xlabel("x (m)")
    #axes[3].set_ylabel("y (m)")


    # Set the correct aspect ratio
    for ax in axes:
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()
    fig.savefig("propagation_smaller_more.pdf")


