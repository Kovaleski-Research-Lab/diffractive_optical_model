#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import torch
import numpy as np
from loguru import logger
import torchvision
import pytorch_lightning as pl

#from . import plane
import plane


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
        #   Shift: Shifted angular spectrum method
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

        # The scaled propagation is a special case of the ASM and RSC propagation
        # methods where the chirp z transform is used to perform the IFFT. This
        # scales the output appropriately.
        ###

        # Check: Are the input and output planes tilted? If so, create a rotation
        # matrix to rotate both planes to the same orientation.
        if (input_plane.normal != output_plane.normal).any():
            logger.debug("Input and output planes are tilted. Creating rotation matrix")
            # This is not used yet
            rot = self.create_rotation_matrix(input_plane.normal, output_plane.normal)
        
        # If the wavelength is not a tensor, convert it to a tensor.
        if not torch.is_tensor(params['wavelength']):
            wavelength = torch.tensor(params['wavelength'])
        else:
            wavelength = params['wavelength']

        # If the CZT flag is not set, set it to False.
        czt = params['czt']
        if czt == None or czt == False or czt == 'None':
            czt = False
        # Check if the size and discretization of the input and output planes are
        # the same. If they aren't, set the CZT flag to True.
        elif (input_plane.Nx != output_plane.Nx) or (input_plane.Ny != output_plane.Ny):
                logger.debug("Input and output planes have different sizes. Using CZT")
                czt = True

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


        propagator = Propagator(input_plane, output_plane, prop_function, prop_type, czt, wavelength)
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

        # Get the fourier dimensions of the input plane
        fxx, fyy = input_plane.fxx_padded, input_plane.fyy_padded

        #Mask out non-propagating waves
        mask = torch.sqrt(fxx**2 + fyy**2) < (1/wavelength)
        fxx = mask * fxx
        fyy = mask * fyy
        
        #10.1364/JOSAA.401908 equation 28
        #Also Goodman eq 3-78
        fz = torch.sqrt(1 - (wavelength*fxx)**2 - (wavelength*fyy)**2).double()
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
        H = H * torch.exp(1j * 2 * torch.pi * (fxx * x_shift + fyy * y_shift))
        H = torch.fft.fftshift(H)

        H.requrires_grad = False
        return H

    def init_rsc_transfer_function(self, input_plane, output_plane, wavelength):
        logger.debug("Initializing RSC transfer function")

        # Get the spatial dimensions of the input plane
        xx, yy = input_plane.xx_padded, input_plane.yy_padded

        # Get the axial distance between the input and output planes.
        # NOTE: When we move to rotation propatation, this will need to be updated
        # to get the distance between the planes along the propagation axis.
        #distance = torch.norm(output_plane.center - input_plane.center)
        distance = output_plane.center[-1] - input_plane.center[-1]

        #10.1364/JOSAA.401908 equation 29
        #Also Goodman eq 3-79
        r = torch.sqrt(xx**2 + yy**2 + distance**2).double()
        k = (2 * torch.pi / wavelength).double()
        z = distance.double()

        # Initialize the impulse response
        h_rsc = torch.exp(1j*k*r) / r
        h_rsc *= ((1/r) - (1j*k))
        h_rsc *= (1/(2*torch.pi)) * (z/r)

        # Get the transfer function
        H = torch.fft.fft2(h_rsc)

        # Normalize the transfer function
        mag = H.abs()
        ang = H.angle()
        mag = mag / torch.max(mag)
        H = mag * torch.exp(1j*ang)
        
        # Get the fourier dimensions of the input plane
        fxx, fyy = input_plane.fxx_padded, input_plane.fyy_padded

        # Get the x-y shift between the input and output planes.
        shift = output_plane.center - input_plane.center
        x_shift = shift[0]
        y_shift = shift[1]

        # Shift the transfer function to account for the shift between the input and output planes.
        H = H * torch.exp(1j * 2 * torch.pi * (fxx * x_shift + fyy * y_shift))

        H.requrires_grad = False
        return H

class Propagator(pl.LightningModule):
    def __init__(self, input_plane, output_plane, transfer_function, prop_type, czt, wavelength):
        super().__init__()
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.transfer_function = transfer_function
        self.prop_type = prop_type
        self.wavelength = wavelength
        self.register_buffer('H', self.transfer_function)
        self.cc_output = torchvision.transforms.CenterCrop((int(output_plane.Nx), int(output_plane.Ny)))
        self.cc_input = torchvision.transforms.CenterCrop((int(input_plane.Nx), int(input_plane.Ny)))
        padx = torch.div(input_plane.Nx, 2, rounding_mode='trunc')
        pady = torch.div(input_plane.Ny, 2, rounding_mode='trunc')
        self.padding = (pady,pady,padx,padx)    

        self.czt = czt
        # Initialize the CZT if needed  
        if self.czt:
            self.init_czt()

    def normalize(self, x):
        mag = x.abs()
        ang = x.angle()
        mag = mag - torch.min(mag)
        mag = mag / torch.max(mag)
        return mag * torch.exp(1j*ang)

    def normalize_numpy(self, x):
        mag = np.abs(x)
        ang = np.angle(x)
        mag = mag - np.min(mag)
        mag = mag / np.max(mag)
        return mag * np.exp(1j*ang)

    def init_czt(self):

        # Equations and notation from 10.1364/JOSAA.31.001832
        dx_d = self.output_plane.delta_x.numpy()
        dy_d = self.output_plane.delta_y.numpy()

        dfx = self.input_plane.delta_fx_padded.numpy()
        dfy = self.input_plane.delta_fy_padded.numpy()

        xx_d = self.output_plane.xx.numpy()
        yy_d = self.output_plane.yy.numpy()

        # Scale factors
        self.alpha_x = np.round(dx_d/dfx, 10)
        self.alpha_y = np.round(dy_d/dfy, 10)

        # New coordinates
        wx = self.alpha_x*self.input_plane.fx_padded.numpy()
        wy = self.alpha_y*self.input_plane.fy_padded.numpy()
        wxx, wyy = np.meshgrid(wx, wy)

        self.dwx = np.round(np.diff(wx)[0], 10)
        self.dwy = np.round(np.diff(wy)[0], 10)

        assert np.allclose(self.dwx,dx_d), "dx_d = {} and dwx = {}".format(dx_d,self.dwx)
        assert np.allclose(self.dwy,dy_d), "dy_d = {} and dwy = {}".format(dy_d,self.dwy)

        C = np.exp(1j * np.pi * ((xx_d**2)/(self.alpha_x) + (yy_d**2)/(self.alpha_y)))
        D = np.exp(-1j * np.pi * ((wxx**2)/(self.alpha_x) + (wyy**2)/(self.alpha_y)))
        E = np.exp(1j * np.pi * ((wxx**2)/(self.alpha_x) + (wyy**2)/(self.alpha_y)))

        #C = self.normalize_numpy(C)
        #D = self.normalize_numpy(D)
        #E = self.normalize_numpy(E)

        D = np.fft.fftshift(D)
        E = np.fft.fftshift(E)

        self.common_x = C.shape[-2] + D.shape[-2] - 1
        self.common_y = C.shape[-1] + D.shape[-1] - 1

        d_padx = (self.common_x - D.shape[-2])//2
        d_pady = (self.common_y - D.shape[-1])//2

        u_padx = (self.common_x - self.input_plane.Nx)//2
        u_pady = (self.common_y - self.input_plane.Ny)//2

        self.d_pad = (d_padx, d_padx, d_pady, d_pady)
        self.u_pad = (u_padx, u_padx, u_pady, u_pady)

        #E = torch.nn.functional.pad(E, self.d_pad, mode='constant')
        #D = torch.nn.functional.pad(D, self.d_pad, mode='circular')

        E = np.pad(E, [(d_padx,d_padx), (d_pady,d_pady)], mode='constant')
        D = np.pad(D, [(d_padx,d_padx), (d_pady,d_pady)], mode='wrap')

        C = torch.from_numpy(C)
        D = torch.from_numpy(D)
        E = torch.from_numpy(E)

        C = C.unsqueeze(0).unsqueeze(0)
        D = D.unsqueeze(0).unsqueeze(0)
        E = E.unsqueeze(0).unsqueeze(0)

        self.register_buffer('C', C)
        self.register_buffer('D', D)
        self.register_buffer('E', E)

    def czt_ifft(self, Uz):
        
        Uz = self.cc_input(Uz)
        
        Uz = torch.nn.functional.pad(Uz, self.u_pad, mode='constant')

        # Scale Uz - they call it U^z_w in the paper
        Uzw = Uz * self.E / (self.alpha_x*self.alpha_y)

        #Uzw = np.pad(Uzw, [(u_padx,u_padx), (u_padx,u_pady)], mode='constant')

        # Linear convolution of Uzw with D
        R = torch.fft.fft2(Uzw)
        if self.prop_type == 'asm':
            R = torch.fft.fftshift(R)

        S = torch.fft.fft2(self.D)
        Uzw_d = torch.fft.ifft2(R * S)
        Uzw_d = torch.fft.fftshift(Uzw_d, dim=(-1,-2))

        # Crop the result
        Uzw_d = torch.nn.functional.pad(Uzw_d, (1, 1, 1, 1), mode='constant')
        Uzw_d = self.cc_output(Uzw_d)

        ## Scale the result
        uz = Uzw_d * self.C * self.dwx * self.dwy
        return uz

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

        return self.normalize(self.cc_output(output_wavefront))

    def asm_propagate(self, input_wavefront):
        #logger.debug("Propagating using ASM")
        ###
        # Propagates the wavefront using the angular spectrum method.
        ###
        A = torch.fft.fft2(input_wavefront)
        A = torch.fft.fftshift(A, dim=(-1,-2))
        U = A * self.H

        if self.czt:
            U = self.czt_ifft(U)
            #U = torch.fft.ifftshift(U, dim=(-1,-2))
        else:
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

        if self.czt:
            U = torch.fft.fftshift(U, dim=(-1,-2))
            U = self.czt_ifft(U)
        else:
            U = torch.fft.ifft2(U)
            U = torch.fft.ifftshift(U, dim=(-1,-2))
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
        'center': torch.tensor([0.,0.,10e-2])
    }

    output_plane_params1 = {
        'name': 'output_plane',
        'size': torch.tensor([8.96e-3, 8.96e-3]),
        'Nx': 1080,
        'Ny': 1080,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0.,0.,10e-2])
    }

    input_plane = plane.Plane(input_plane_params)
    output_plane0 = plane.Plane(output_plane_params0)
    output_plane1 = plane.Plane(output_plane_params1)

    propagator_params = {
        'prop_type': None,
        'wavelength': torch.tensor(1.55e-6),
        'czt': False
    }

    propagator0 = PropagatorFactory()(input_plane, output_plane0, propagator_params)

    propagator_params['prop_type'] = None
    propagator_params['czt'] = True
    propagator1 = PropagatorFactory()(input_plane, output_plane1, propagator_params)

    # Example wavefront to propagate
    # This is a plane wave through a 1mm aperture
    x = torch.linspace(-input_plane.Lx/2, input_plane.Lx/2, input_plane.Nx)
    y = torch.linspace(-input_plane.Ly/2, input_plane.Ly/2, input_plane.Ny)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    wavefront = torch.ones_like(xx)
    wavefront[(xx**2 + yy**2) > (0.2e-3)**2] = 0
    wavefront = wavefront.view(1,1,input_plane.Nx,input_plane.Ny)

    # Propagate the wavefront
    output_wavefront0 = propagator0(wavefront)
    output_wavefront1 = propagator1(wavefront)
    #output_wavefront2 = propagator2(wavefront)

    difference = np.abs(output_wavefront0.abs() - output_wavefront1.abs())

    # Plot the input and output wavefronts
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1,4)
    im0 = axes[0].imshow(wavefront[0,0,:,:].abs().numpy(), vmin=0, vmax=1)
    axes[0].set_title("Input wavefront")
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax)

    im1 = axes[1].imshow(output_wavefront0[0,0,:,:].abs().numpy(), vmin=0, vmax=1)
    axes[1].set_title("Output wavefront (no CZT)")
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)

    im2 = axes[2].imshow(output_wavefront1[0,0,:,:].abs().numpy(), vmin=0, vmax=1)
    axes[2].set_title("Output wavefront (CZT)")
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)

    im3 = axes[3].imshow(difference[0,0,:,:], vmin=0, vmax=1)
    axes[3].set_title("Difference")
    divider = make_axes_locatable(axes[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax)

    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("y (m)")
    axes[3].set_xlabel("x (m)")
    axes[3].set_ylabel("y (m)")


    # Set the correct aspect ratio
    for ax in axes:
        ax.set_aspect('equal')

    plt.show()


