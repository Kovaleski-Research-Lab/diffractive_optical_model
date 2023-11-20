#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import torch
import torchvision
from loguru import logger
import pytorch_lightning as pl

import plane

#--------------------------------
# Initialize: Propagator
#--------------------------------

class oldPropagator(pl.LightningModule):
    def __init__(self, params, paths):
        super().__init__()
        logger.debug("Initializing Propagator")

        self.params = params.copy()
        self.paths = paths.copy()

        # Load: Physical parameters
        self.Lxp = torch.tensor(self.params['Lxp'])   
        self.Lyp = torch.tensor(self.params['Lyp'])   
        self.Nxp = torch.tensor(self.params['Nxp'])
        self.Nyp = torch.tensor(self.params['Nyp'])
        self.distance = torch.tensor(params['distance'])
        logger.debug("Setting distance to {}".format(self.distance))
        self.wavelength = torch.tensor(self.params['wavelength'])
        logger.debug("Setting wavelength to {}".format(self.wavelength))
        self.wavenumber = 2 * torch.pi / self.wavelength 

        self.delta_x = self.Lxp / self.Nxp
        self.delta_y = self.Lyp / self.Nyp
        logger.debug("Setting sampling pitch {}x{}".format(self.delta_x, self.delta_y))

        # Initialize: Center crop transform
        self.cc = torchvision.transforms.CenterCrop((int(self.Nxp), int(self.Nyp)))

        #Create: The propagator
        self.asm = None
        self.adaptive = self.params['adaptive']
        logger.debug("Setting adaptive to {}".format(self.adaptive))

        self.create_propagator()
 
    def create_propagator(self):
        logger.debug("Creating propagation layer")
        padx = torch.div(self.Nxp, 2, rounding_mode='trunc')
        pady = torch.div(self.Nyp, 2, rounding_mode='trunc')
        self.padding = (pady,pady,padx,padx)    

        if self.check_asm_distance():
            self.asm = True
            self.init_asm_transfer_function()
        else:
            self.asm = False
            self.init_rsc_transfer_function()


    def update_propagator(self):
        logger.debug("Updating propgator due to specified distance")
        if self.adaptive:
            if self.check_asm_distance():
                self.asm = True
                self.init_asm_transfer_function()
            else:
                self.asm = False
                self.init_rsc_transfer_function()
        else:
            if self.asm:
                self.init_asm_transfer_function()
            else:
                self.init_rsc_transfer_function()

    def check_asm_distance(self):
        logger.debug("Checking ASM propagation criteria")
        #10.1364/JOSAA.401908 equation 32
        #Checks distance criteria for sampling considerations
        distance_criteria_y = 2 * self.Nyp * (self.delta_y**2) / self.wavelength
        distance_criteria_y *= torch.sqrt(1 - (self.wavelength / (2 * self.Nyp))**2)
       
        distance_criteria_x = 2 * self.Nxp * (self.delta_x**2) / self.wavelength
        distance_criteria_x *= torch.sqrt(1 - (self.wavelength / (2 * self.Nxp))**2)
        
        strict_distance = torch.min(distance_criteria_y, distance_criteria_x) 
        logger.debug("Maximum propagation distance for asm : {}".format(strict_distance))
    
        return(torch.le(self.distance, strict_distance))
 
    #--------------------------------
    # Initialize: ASM transfer fxn
    #--------------------------------

    def init_asm_transfer_function(self): 
        logger.debug("Initializing ASM transfer function")
        self.x = torch.linspace(-self.Lxp / 2, self.Lxp / 2, self.Nxp)
        self.y = torch.linspace(-self.Lyp / 2, self.Lyp / 2, self.Nyp)
        self.xx, self.yy = torch.meshgrid(self.x, self.y, indexing='ij')
        
        #Double the number of samples to eliminate asm errors
        self.fx = torch.fft.fftfreq(2*len(self.x), torch.diff(self.x)[0]).to(self.device)
        self.fy = torch.fft.fftfreq(2*len(self.y), torch.diff(self.y)[0]).to(self.device)
        self.fxx, self.fyy = torch.meshgrid(self.fx, self.fy, indexing='ij')

        #Mask out non-propagating waves
        mask = torch.sqrt(self.fxx**2 + self.fyy**2) < (1/self.wavelength)
        self.fxx = mask * self.fxx
        self.fyy = mask * self.fyy
        
        #10.1364/JOSAA.401908 equation 28
        #Also Goodman eq 3-78
        self.fz = torch.sqrt(1 - (self.wavelength*self.fxx)**2 - (self.wavelength*self.fyy)**2).double()
        self.fz *= ((torch.pi * 2)/self.wavelength).to(self.device)

        H = torch.exp(1j * self.distance * self.fz)
        H = torch.fft.fftshift(H)
        H.requrires_grad = False
        self.register_buffer('H', H)  
 
    #--------------------------------
    # Initialize: RSC transfer fxn
    #--------------------------------

    def init_rsc_transfer_function(self):
        logger.debug("Initializing RSC transfer function")
        #Double the size to eliminate rsc errors.
        self.x = torch.linspace(-self.Lxp , self.Lxp, 2*self.Nxp).to(self.device)
        self.y = torch.linspace(-self.Lyp , self.Lyp, 2*self.Nyp).to(self.device)
        self.xx,self.yy = torch.meshgrid(self.x, self.y, indexing='ij')

        #10.1364/JOSAA.401908 equation 29
        #Also Goodman eq 3-79
        r = torch.sqrt(self.xx**2 + self.yy**2 + self.distance**2).double()
        k = (2 * torch.pi / self.wavelength).double()
        z = self.distance.double()

        h_rsc = torch.exp(1j*k*r) / r
        h_rsc *= ((1/r) - (1j*k))
        h_rsc *= (1/(2*torch.pi)) * (z/r)
        H = torch.fft.fft2(h_rsc)
        mag = H.abs()
        ang = H.angle()
        
        mag = mag / torch.max(mag)
        H = mag * torch.exp(1j*ang)
        H.requrires_grad = False
        self.register_buffer('H', H) 

    #--------------------------------
    # Initialize: Helper to crop
    #--------------------------------

    def center_crop_wavefront(self, wavefront):
        return self.cc(wavefront)
 
    #--------------------------------
    # Initialize: Forward pass
    #--------------------------------

    def forward(self, wavefront, distance = None):

        if distance is not None:
            self.distance = distance
            self.update_propagator()
        if wavefront.shape != self.H.shape:
            wavefront = torch.nn.functional.pad(wavefront,self.padding,mode="constant") 
        # The different methods require a different ordering of the shifts...
        if self.asm:
            A = torch.fft.fft2(wavefront)
            A = torch.fft.fftshift(A, dim=(-1,-2))
            U = A * self.H
            U = torch.fft.ifftshift(U, dim=(-1,-2))
            U = torch.fft.ifft2(U)
        else:
            A = torch.fft.fft2(wavefront)
            U = A * self.H 
            U = torch.fft.ifft2(U)
            U = torch.fft.ifftshift(U, dim=(-1,-2))
        U = self.center_crop_wavefront(U)
        return U


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
        distance = torch.norm(output_plane.center - input_plane.center)
        if self.check_asm_distance(input_plane, output_plane, params):
            logger.debug("Using ASM propagator")
            transfer_function = self.init_asm_transfer_function(input_plane, output_plane, params['wavelength'])
            propagator = Propagator(input_plane, output_plane, transfer_function, 'asm')
        else:
            logger.debug("Using RSC propagator")
            transfer_function = self.init_rsc_transfer_function(input_plane, output_plane, params['wavelength'])
            propagator = Propagator(input_plane, output_plane, transfer_function, 'rsc')

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
        wavelength = params['wavelength']
        delta_x = input_plane.delta_x
        delta_y = input_plane.delta_y
        Nx = input_plane.Nx
        Ny = input_plane.Ny

        distance = torch.norm(output_plane.center - input_plane.center)

        logger.debug("Distance between input and output planes: {}".format(distance))

        distance_criteria_y = 2 * Ny * (delta_y**2) / wavelength
        distance_criteria_y *= torch.sqrt(1 - (wavelength / (2 * Ny))**2)
       
        distance_criteria_x = 2 * Nx * (delta_x**2) / wavelength
        distance_criteria_x *= torch.sqrt(1 - (wavelength / (2 * Nx))**2)
        
        strict_distance = torch.min(distance_criteria_y, distance_criteria_x) 
        logger.debug("Maximum propagation distance for asm : {}".format(strict_distance))
    
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

        H = torch.exp(1j * distance * fz)
        mag = H.abs()
        ang = H.angle()
        mag = mag / torch.max(mag)
        H = mag * torch.exp(1j*ang)
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
        mag = H.abs()
        ang = H.angle()
        
        mag = mag / torch.max(mag)
        H = mag * torch.exp(1j*ang)
        H.requrires_grad = False
        return H

class Propagator(pl.LightningModule):
    def __init__(self, input_plane, output_plane, transfer_function, prop_type='asm'):
        super().__init__()
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.transfer_function = transfer_function
        self.prop_type = prop_type
        self.H = transfer_function
        self.cc = torchvision.transforms.CenterCrop((int(input_plane.Nx), int(input_plane.Ny)))
        padx = torch.div(input_plane.Nx, 2, rounding_mode='trunc')
        pady = torch.div(input_plane.Ny, 2, rounding_mode='trunc')
        self.padding = (pady,pady,padx,padx)    

    def forward(self, input_wavefront):
        # Pad the wavefront
        input_wavefront = torch.nn.functional.pad(input_wavefront,self.padding,mode="constant")
        return self.rotate(self.shift(self.propagate(self.rotate(input_wavefront))))

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
        logger.debug("Propagating using ASM")
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
        logger.debug("Propagating using RSC")
        ###
        # Propagates the wavefront using the rayleigh-sommerfeld convolution.
        ###
        A = torch.fft.fft2(input_wavefront)
        U = A * self.H 
        U = torch.fft.ifft2(U)
        U = torch.fft.ifftshift(U, dim=(-1,-2))
        U = U 
        return U

    def shift(self, input_wavefront):
        ###
        # Uses the shift property of the fourier transform to shift the wavefront.
        ###
        logger.warning("Shift not implemented")
        return input_wavefront

    def rotate(self, input_wavefront):
        ###
        # Rotates the wavefront to the orientation of the output plane.
        ###
        logger.warning("Rotate not implemented")
        return input_wavefront

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
        'center': torch.tensor([0,0,9.6e-2])
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
        'center': torch.tensor([0,0,9.61e-2])
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
    fig, axes = plt.subplots(1,3)
    axes[0].imshow(wavefront[0,0,:,:].abs().numpy()**2)
    axes[0].set_title("Input wavefront")
    axes[1].imshow(output_wavefront0[0,0,:,:].abs().numpy()**2)
    axes[1].set_title("Output wavefront 0")
    axes[2].imshow(output_wavefront1[0,0,:,:].abs().numpy()**2)
    axes[2].set_title("Output wavefront 1")

    plt.show()


