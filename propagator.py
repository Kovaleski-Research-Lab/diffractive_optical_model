#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import torch
from loguru import logger
import torchvision
import pytorch_lightning as pl

from . import plane

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
        #   ASM: Angular spectrum method
        #   RSC: Rayleigh-Sommerfeld convolution
        #   Shift: Shifted angular spectrum method
        #   Scaled: Scaled angular spectrum method 
        #   Rotated: Rotated angular spectrum method

        # Propagator references:
        #   ASM: 10.1364/JOSAA.401908 
        #   RSC: 10.1364/JOSAA.401908
        #   Shift: 10.1364/OE.18.018453 
        #   Scaled: 10.1364/OL.37.004128 
        #   Rotated: 10.1364/JOSAA.20.001755

        # This function compares the input and output planes to determine 
        # which propagator type to use and in which order to apply the propagators.

        # The order of operations is:
        #   1. Rotate
        #   2. Propagate
        #      a. ASM or RSC?
        #      c. Scaled?
        #   3. Shift
        #   4. Rotate

        # The scaled propagation is a special case of the ASM and RSC propagation
        # methods where the nonuniform fourier transform is used to scale the
        # propagation planes. The choice between ASM and RSC is determined by the
        # distance between the input and output planes.
        ###

        # Check: Are the input and output planes tilted? If so, create a rotation
        # matrix to rotate both planes to the same orientation.
        if (input_plane.normal != output_plane.normal).any():
            logger.debug("Input and output planes are tilted. Creating rotation matrix")
            rot = self.create_rotation_matrix(input_plane.normal, output_plane.normal)
        
        # Check: The distance between the centers of input and output planes. If the distance
        # is less than the ASM distance, use the RSC propagator. Otherwise, use
        # the ASM propagator.
        wavelength = torch.tensor(params['wavelength'])
        distance = torch.norm(output_plane.center - input_plane.center)
        if self.check_asm_distance(input_plane, output_plane, params):
            logger.debug("Using ASM propagator")
            transfer_function = self.init_asm_transfer_function(input_plane, output_plane, wavelength)
            prop_type = 'asm'
            propagator = Propagator(input_plane, output_plane, transfer_function, 'asm')
        else:
            logger.debug("Using RSC propagator")
            prop_type = 'rsc'
            transfer_function = self.init_rsc_transfer_function(input_plane, output_plane, wavelength)

        propagator = Propagator(input_plane, output_plane, transfer_function, prop_type)
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

        logger.debug("Axial distance between input and output planes: {}".format(distance))

        distance_criteria_y = 2 * delta_y * ( Ny * delta_y - shift_y) / wavelength
        distance_criteria_y *= torch.sqrt(1 - (wavelength / (2 * Ny))**2)
       
        distance_criteria_x = 2 * delta_x * ( Nx * delta_x - shift_x) / wavelength
        distance_criteria_x *= torch.sqrt(1 - (wavelength / (2 * Nx))**2)
        
        strict_distance = torch.min(distance_criteria_y, distance_criteria_x) 
        logger.debug("Maximum axial distance for asm : {}".format(strict_distance))
    
        return(torch.le(distance, strict_distance))

    def init_asm_transfer_function(self, input_plane, output_plane, wavelength): 
        logger.debug("Initializing ASM transfer function")

        #Double the number of samples to eliminate asm errors
        fx = torch.fft.fftfreq(2*len(input_plane.x), torch.diff(input_plane.x)[0])
        fy = torch.fft.fftfreq(2*len(input_plane.y), torch.diff(input_plane.y)[0])
        fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')

        #Mask out non-propagating waves
        mask = torch.sqrt(fxx**2 + fyy**2) < (1/wavelength)
        fxx = mask * fxx
        fyy = mask * fyy
        
        #10.1364/JOSAA.401908 equation 28
        #Also Goodman eq 3-78
        fz = torch.sqrt(1 - (wavelength*fxx)**2 - (wavelength*fyy)**2).double()
        fz *= ((torch.pi * 2)/wavelength)

        # Get the distance between the input and output planes.
        distance = torch.norm(output_plane.center - input_plane.center)

        # Get the x-y shift between the input and output planes.
        shift = output_plane.center - input_plane.center
        x_shift = shift[0]
        y_shift = shift[1]

        H = torch.exp(1j * distance * fz)

        # Normalize the transfer function
        mag = H.abs()
        ang = H.angle()
        mag = mag / torch.max(mag)
        H = mag * torch.exp(1j*ang)

        # Shift the transfer function to account for the shift between the input and output planes.
        H = H * torch.exp(1j * 2 * torch.pi * (fxx * x_shift + fyy * y_shift))
        H = torch.fft.fftshift(H)

        H.requrires_grad = False
        return H

    def init_rsc_transfer_function(self, input_plane, output_plane, wavelength):
        logger.debug("Initializing RSC transfer function")
        #Double the size to eliminate rsc errors.
        x = torch.linspace(-input_plane.Lx , input_plane.Lx, 2*input_plane.Nx)
        y = torch.linspace(-input_plane.Ly , input_plane.Ly, 2*input_plane.Ny)
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        # Get the distance between the input and output planes.
        distance = torch.norm(output_plane.center - input_plane.center)

        #10.1364/JOSAA.401908 equation 29
        #Also Goodman eq 3-79
        r = torch.sqrt(xx**2 + yy**2 + distance**2).double()
        k = (2 * torch.pi / wavelength).double()
        z = distance.double()

        h_rsc = torch.exp(1j*k*r) / r
        h_rsc *= ((1/r) - (1j*k))
        h_rsc *= (1/(2*torch.pi)) * (z/r)
        H = torch.fft.fft2(h_rsc)
        
        # Get the fourier dimensions
        fx = torch.fft.fftfreq(len(x), torch.diff(x)[0])
        fy = torch.fft.fftfreq(len(y), torch.diff(y)[0])
        fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')

        # Get the x-y shift between the input and output planes.
        shift = output_plane.center - input_plane.center
        x_shift = shift[0]
        y_shift = shift[1]

        # Normalize the transfer function
        mag = H.abs()
        ang = H.angle()
        mag = mag / torch.max(mag)
        H = mag * torch.exp(1j*ang)

        # Shift the transfer function to account for the shift between the input and output planes.
        H = H * torch.exp(1j * 2 * torch.pi * (fxx * x_shift + fyy * y_shift))

        H.requrires_grad = False
        return H

class Propagator(pl.LightningModule):
    def __init__(self, input_plane, output_plane, transfer_function, prop_type='asm'):
        super().__init__()
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.transfer_function = transfer_function
        self.prop_type = prop_type
        self.register_buffer('H', self.transfer_function)
        self.cc = torchvision.transforms.CenterCrop((int(input_plane.Nx), int(input_plane.Ny)))
        padx = torch.div(input_plane.Nx, 2, rounding_mode='trunc')
        pady = torch.div(input_plane.Ny, 2, rounding_mode='trunc')
        self.padding = (pady,pady,padx,padx)    

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
        else:
            logger.error("Invalid propagation type: {}".format(self.prop_type))
            raise ValueError("Invalid propagation type: {}".format(self.prop_type))
        return self.cc(output_wavefront)

    def asm_propagate(self, input_wavefront):
        #logger.debug("Propagating using ASM")
        ###
        # Propagates the wavefront using the angular spectrum method.
        ###
        A = torch.fft.fft2(input_wavefront)
        A = torch.fft.fftshift(A, dim=(-1,-2))
        U = A * self.H
        U = torch.fft.ifftshift(U, dim=(-1,-2))
        U = torch.fft.ifft2(U)
        return U

    def rsc_propagate(self, input_wavefront):
        #logger.debug("Propagating using RSC")
        ###
        # Propagates the wavefront using the rayleigh-sommerfeld convolution.
        ###
        A = torch.fft.fft2(input_wavefront)
        U = A * self.H 
        U = torch.fft.ifft2(U)
        U = torch.fft.ifftshift(U, dim=(-1,-2))
        U = U 
        return U

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
        'Nx': 1080,
        'Ny': 1080,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0,0,0])
    }

    output_plane_params0 = {
        'name': 'output_plane',
        'size': torch.tensor([8.96e-3, 8.96e-3]),
        'Nx': 1080,
        'Ny': 1080,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([2.e-3,2.e-3,9.6e-2])
    }

    propagator_params = {
            'wavelength': torch.tensor(1.55e-6),
            }

    input_plane = plane.Plane(input_plane_params)
    output_plane0 = plane.Plane(output_plane_params0)

    output_plane_params1 = {
        'name': 'output_plane',
        'size': torch.tensor([8.96e-3, 8.96e-3]),
        'Nx': 1080,
        'Ny': 1080,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([2.e-3,2.e-3,9.61e-2])
    }

    output_plane1 = plane.Plane(output_plane_params1)

    propagator_params = {
        'wavelength': torch.tensor(1.55e-6),
    }

    propagator0 = PropagatorFactory()(input_plane, output_plane0, propagator_params)
    propagator1 = PropagatorFactory()(input_plane, output_plane1, propagator_params)

    # Example wavefront to propagate
    # This is a plane wave through a 1mm aperture
    x = torch.linspace(-input_plane.Lx/2, input_plane.Lx/2, input_plane.Nx)
    y = torch.linspace(-input_plane.Ly/2, input_plane.Ly/2, input_plane.Ny)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    wavefront = torch.ones_like(xx)
    wavefront[(xx**2 + yy**2) > (1e-3)**2] = 0
    wavefront = wavefront.view(1,1,input_plane.Nx,input_plane.Ny)

    # Propagate the wavefront
    output_wavefront0 = propagator0(wavefront)
    output_wavefront1 = propagator1(wavefront)

    # Plot the input and output wavefronts
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    axes[0].pcolormesh(xx.numpy(), yy.numpy(), wavefront[0,0,:,:].abs().numpy())
    axes[0].set_title("Input wavefront")
    axes[1].pcolormesh(xx.numpy(), yy.numpy(), output_wavefront0[0,0,:,:].abs().numpy())
    axes[1].set_title("Output wavefront for ASM\n x0 = 2mm, y0 = 2mm, z = 9.6cm")
    axes[2].pcolormesh(xx.numpy(), yy.numpy(), output_wavefront1[0,0,:,:].abs().numpy())
    axes[2].set_title("Output wavefront for RSC\n x0 = 2mm, y0 = 2mm, z = 9.61cm")

    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("y (m)")


    # Set the correct aspect ratio
    for ax in axes:
        ax.set_aspect('equal')

    plt.show()


