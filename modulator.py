#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
import os
import sys
import torch
import logging
from IPython import embed
from loguru import logger
import pytorch_lightning as pl

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

#from . import plane
import plane

#--------------------------------
# Initialize: Wavefront Modulator
#--------------------------------

class oldModulator(pl.LightningModule):
    def __init__(self, params, paths):
        super().__init__()
        logging.debug("modulator.py - Initializing Modulator")
        # Load : Copy of parameters 
        self.params = params.copy()
        self.paths = paths.copy()

        # Load : General modulator parameters 
        self.gradients = self.params['gradients']
        self.phase_initialization = self.params['phase_initialization']
        self.amplitude_initialization = self.params['amplitude_initialization']

        # Load : Physical parameters 
        self.Lxm = torch.tensor(self.params['Lxm'])
        self.Lym = torch.tensor(self.params['Lym'])
        self.Nxm = torch.tensor(self.params['Nxm'])
        self.Nym = torch.tensor(self.params['Nym'])
        self.wavelength = torch.tensor(self.params['wavelength'])
       
        # Load : Calibration params
        self.calibrate_phase = params['calibrate_phase']
        self.calibrate_amplitude = params['calibrate_amplitude']

        self.load_calibration = params['load_calibration']

        self.path_calibration = self.params['path_calibration']
        self.calibration_name = self.params['calibration_name']
        self.path_root = self.paths['path_root']
        self.path_results = self.paths['path_results']

        self.path_save = os.path.join(self.path_root, self.path_results, self.path_calibration)
        # Create : The modulator
        self.create_modulator()

    #--------------------------------
    # Initialize : Getters
    #--------------------------------

    def get_phase(self, with_grad=True):
        if with_grad:
            return self.phase
        else:
            return self.phase.detach()

    def get_amplitude(self, with_grad=True):
        if with_grad:
            return self.amplitude
        else:
            return self.amplitude.detach()

    #--------------------------------
    # Initialize : Setters
    #--------------------------------

    def set_amplitude(self, amplitude):
        amplitude = torch.tensor(amplitude).view(1,1,self.Nxm, self.Nym)
        self.amplitude = torch.nn.Parameter(amplitude).to(torch.double)
        self.initialize_gradients()

    def set_phase(self, phase):
        phase = torch.tensor(phase).view(1,1,self.Nxm, self.Nym)
        self.phase = torch.nn.Parameter(phase).to(torch.double)
        self.initialize_gradients()

    def save_calibration(self):
        logging.debug("modulator.py | Saving calibration")
        assert (self.calibrate_amplitude or self.calibrate_phase)
        if os.path.exists(os.path.join(self.path_save, self.calibration_name)):
            temp_amp, temp_phase = torch.load(os.path.join(self.path_save, self.calibration_name))
            if self.calibrate_phase:
                phase = self.cal_phase.detach().cpu()
                amp = temp_amp
            if self.calibrate_amplitude:
                amp = self.cal_amp.detach().cpu()
                phase = temp_phase
        else:
            os.makedirs(self.path_save, exist_ok=True)
            amp,phase = self.cal_amp.detach(), self.cal_phase.detach()
            amp = amp.cpu()
            phase = phase.cpu()

        torch.save([amp, phase], os.path.join(self.path_save, self.calibration_name))

    #--------------------------------
    # Create: Wavefront Modulator
    #--------------------------------

    def create_modulator(self):
        # Initialize : Meshgrid for phase and amplitude values
        self.x = torch.linspace(-self.Lxm / 2 , self.Lxm / 2, self.Nxm)
        self.y = torch.linspace(-self.Lym / 2 , self.Lym / 2, self.Nym)
        self.xx , self.yy = torch.meshgrid(self.x, self.y, indexing='ij')
    
        # Initialize : Amplitude and phase values
        amplitude = self.init_amplitude()
        phase = self.init_phase()

        # Register : Amplitude and phase values
        self.register_parameter('amplitude', amplitude)
        self.register_parameter('phase', phase)

        #Initialize : Calibration parameters
        if self.calibrate_amplitude:
            temp_cal_amp = torch.nn.Parameter(torch.rand(amplitude.shape))
        else:
            temp_cal_amp = torch.nn.Parameter(torch.zeros(amplitude.shape))
        if self.calibrate_phase:
            temp_cal_phase = torch.nn.Parameter(torch.rand(phase.shape))
        else:
            temp_cal_phase = torch.nn.Parameter(torch.zeros(phase.shape))

        # Register : Calibration parameters
        self.register_parameter('cal_amp', temp_cal_amp)
        self.register_parameter('cal_phase', temp_cal_phase) 

        # Check : Are we loading in a calibration?
        if self.load_calibration:
            calibration = torch.load(os.path.join(self.path_save, self.calibration_name))
            cal_amp, cal_phase = calibration
            self.cal_amp = torch.nn.Parameter(cal_amp)
            self.cal_phase = torch.nn.Parameter(cal_phase)

        self.initialize_gradients()

    #--------------------------------
    # Initialize: Parameter Gradients
    #--------------------------------

    def initialize_gradients(self):
        # Initialize : Parameter gradients 
        if self.gradients =='complex':
            logging.debug("Modulator | Keeping all gradients")
            pass
        elif self.gradients == 'phase_only':
            logging.debug("Modulator | Setting amplitude.requires_grad to False")
            self.amplitude.requires_grad = False
        elif self.gradients == 'amplitude_only':
            logging.debug("Modulator | Setting phase.requires_grad to False")
            self.phase.requires_grad = False
        elif self.gradients == 'none':
            logging.debug("Modulator | Setting phase.requires_grad to False and amplitude.requires_grad to False")
            self.amplitude.requires_grad = False
            self.phase.requires_grad = False
        else:
            logging.error("Type : {} not implemented!".format(self.type))
            exit() 

        # Are we calibrating? This overwrites the previous checks
        if self.calibrate_phase:
            logging.debug("Modulator | Calibrating phase")
            self.cal_phase.requries_grad = True
            self.cal_amp.requires_grad = False
            self.amplitude.requires_grad = False
            self.phase.requires_grad = False
        else:
            self.cal_phase.requires_grad = False

        if self.calibrate_amplitude:
            logging.debug("Modulator | Calibrating amplitude")
            self.cal_amp.requires_grad = True
            self.cal_phase.requires_grad = False
            self.amplitude.requires_grad = False
            self.phase.requires_grad = False
        else:
            self.cal_amp.requires_grad = False

        if self.calibrate_amplitude and self.calibrate_phase:
            logging.error("You cant calibrate both")
            exit()

    #--------------------------------
    # Initialize: Amplitudes 
    #--------------------------------

    def init_amplitude(self) -> torch.nn.Parameter:
        logging.debug("Modulator | setting amplitude initialization to torch.ones()")
        if self.amplitude_initialization == 'uniform':
            amplitude = torch.nn.Parameter(torch.ones(1,1,self.Nxm, self.Nym).to(torch.float64))
        elif self.amplitude_initialization == 'random':
            amplitude = torch.nn.Parameter(torch.rand(1,1,self.Nxm, self.Nym).to(torch.float64))
        else:
            amplitude = torch.nn.Parameter(torch.ones(1,1,self.Nxm, self.Nym).to(torch.float64))
        return amplitude

    #--------------------------------
    # Initialize: Phases
    #--------------------------------

    def init_phase(self) -> torch.nn.Parameter:
        phase = None
        if self.phase_initialization == "uniform":
            logging.debug("Modulator | setting phase initialization to torch.ones()")
            phase = torch.nn.Parameter(torch.ones(1,1,self.Nxm, self.Nym).to(torch.float64))
        elif self.phase_initialization == "random":
            logging.debug("Modulator | setting phase initialization to torch.rand()")
            phase = torch.nn.Parameter(torch.rand(1,1,self.Nxm, self.Nym).to(torch.float64))
        elif self.phase_initialization == "lens":
            self.focal_length = torch.tensor(self.params['focal_length'])
            phase = -(self.xx**2 + self.yy**2) / ( 2 * self.focal_length )
            phase *= (2 * torch.pi / self.wavelength)
            phase = phase.view(1,1,self.Nxm,self.Nym)
            phase = torch.nn.Parameter(phase.to(torch.float64))

        return phase

    #--------------------------------
    # Reshape : Incident wavefront
    #--------------------------------

    def adjust_shape(self, wavefront):
        shape = wavefront.shape
        w_nx, w_ny = shape[-2:]

        # Assert : Wavefront and modulator pixels are integer multiples
        assert w_nx % self.Nxm == 0 , "Nx and Wnx not divisible"
        assert w_ny % self.Nym == 0 , "Ny and Wny not divisible"
        
        nx_blocks = w_nx // self.Nxm
        ny_blocks = w_ny // self.Nym
       
        block_x = w_nx // nx_blocks
        block_y = w_ny // ny_blocks

        m = torch.nn.Upsample(scale_factor = (nx_blocks, ny_blocks), mode='nearest')
        
        amplitude = m(self.amplitude)
        phase = m(self.phase)

        cal_amp = m(self.cal_amp)
        cal_phase = m(self.cal_phase)
  
        return amplitude,phase, cal_amp, cal_phase


    #--------------------------------
    # Initialize : Forward pass 
    #--------------------------------
    #TODO: we might want to look at different normalization strategies
    def normalize_amplitude(self, amplitude, cal_amp):
        cal_amp = torch.nn.functional.relu(cal_amp)
        amplitude = cal_amp + amplitude
        amplitude = amplitude / torch.max(amplitude)
        return amplitude
            
    #--------------------------------
    # Initialize : Forward pass 
    #--------------------------------

    def forward(self, wavefront):
        if wavefront.squeeze().shape != self.amplitude.squeeze().shape : # type: ignore
            amplitude,phase,cal_amp,cal_phase = self.adjust_shape(wavefront)
        else:
            amplitude,phase,cal_amp,cal_phase = self.amplitude,self.phase,self.cal_amp,self.cal_phase

        amplitude = self.normalize_amplitude(amplitude, cal_amp)
        phase = phase + cal_phase
        
        layer = amplitude * torch.exp(1j*phase)
        return layer * wavefront

#--------------------------------
# Initialize: Lens Modulator
#--------------------------------

#class Lens(Modulator):
#    def __init__(self, params, focal_length):
#        super().__init__(params)
#        logging.debug("modulator.py - Initializing Lens")
#        self.focal_length = focal_length
#        logging.debug("Lens | Setting focal length to {}".format(self.focal_length))
#        self.update_phases()
#
#    def update_phases(self):
#        #----------------------------------
#        #          2pi     - ( x^2 + y^2 )
#        # phase = ------ * ---------------
#        #         lambda          2f
#        #----------------------------------
#        logging.debug("Lens | Updating phases to lens pattern")
#        phase = -(self.xx**2 + self.yy**2) / ( 2 * self.focal_length )
#        phase *= (2 * torch.pi / self.wavelength)
#        phase = phase.view(1,1,self.Nxm,self.Nym)
#        self.phase = torch.nn.Parameter(phase)
#
#        self.initialize_gradients()


class ModulatorFactory():
    def __call__(self, plane, params):
        return self.create_modulator(plane, params)

    def create_modulator(self, plane, params):
        logger.debug("Creating modulator")
        gradients = params['gradients']
        phase_init = params['phase_init']
        amplitude_init = params['amplitude_init']
        phase_pattern = params['phase_pattern']
        amplitude_pattern = params['amplitude_pattern']
        self.kwargs = params['kwargs']

        if phase_init == 'uniform':
            logger.info("Uniform phase initialization")
            phase = self.uniform_phase(plane)
        elif phase_init == 'random':
            logger.info("Random phase initialization")
            phase = self.random_phase(plane)
        elif phase_init == 'custom':
            logger.info("Custom phase initialization")
            phase = self.custom_phase(plane, phase_pattern)
        else:
            logger.warning("Unsupported phase initialization : {}".format(phase_init))
            logger.warning("Setting uniform phase initialization")
            phase = self.uniform_phase(plane)
            raise Exception('unsupportedInitialization')
        
        if amplitude_init == 'uniform':
            logger.info("Uniform amplitude initialization")
            amplitude = self.uniform_amplitude(plane)
        elif amplitude_init == 'random':
            logger.info("Random amplitude initialization")
            amplitude = self.random_amplitude(plane)
        elif amplitude_init == 'custom':
            logger.info("Custom amplitude initialization")
            amplitude = self.custom_amplitude(plane, amplitude_pattern)
        else:
            logger.warning("Unsupported amplitude initialization : {}".format(amplitude_init))
            logger.warning("Setting uniform amplitude initialization")
            amplitude = self.uniform_amplitude(plane)
            raise Exception('unsupportedInitialization')

        if gradients == 'phase_only':
            logger.info("Phase only optimization")
            phase.requires_grad = True
            amplitude.requires_grad = False
        elif gradients == 'amplitude_only':
            logger.info("Amplitude only optimization")
            phase.requires_grad = False
            amplitude.requires_grad = True
        elif gradients == 'complex':
            logger.info("Phase and amplitude optimization")
            phase.requires_grad = True
            amplitude.requires_grad = True
        elif gradients == 'none':
            logger.info("No modulator optimization")
            phase.requires_grad = False
            amplitude.requires_grad = False
        else:
            logger.warning("modulator_type not specified. Setting no modulator optimization")
            phase.requires_grad = False
            amplitude.requires_grad = False
            #Log a non-critical error here

        modulator = Modulator(plane, amplitude.clone(), phase.clone())
        return modulator
    
    #----------------------
    # The assembly line
    #----------------------
    def random_phase(self, plane:plane.Plane) -> torch.Tensor:
        Nx, Ny = plane.Nx, plane.Ny
        phase = torch.rand(1,1,Nx, Ny)
        return phase.to(torch.float64)

    def random_amplitude(self, plane:plane.Plane) -> torch.Tensor:
        Nx, Ny = plane.Nx, plane.Ny
        amplitude = torch.rand(Nx, Ny)
        return amplitude.to(torch.float64)

    def uniform_phase(self, plane:plane.Plane) -> torch.Tensor:
        Nx, Ny = plane.Nx, plane.Ny
        phase = torch.zeros(Nx, Ny)
        return phase.to(torch.float64)

    def uniform_amplitude(self, plane:plane.Plane) -> torch.Tensor:
        Nx, Ny = plane.Nx, plane.Ny
        amplitude = torch.ones(Nx, Ny)
        return amplitude.to(torch.float64)

    def custom_phase(self, plane:plane.Plane, phase_pattern) -> torch.Tensor:
        logger.debug("Creating custom phase")
        # If the phase is a torch tensor
        if isinstance(phase_pattern, torch.Tensor):
            logger.debug("Phase pattern is a custom torch tensor")
            Nx, Ny = plane.Nx, plane.Ny
            phase = phase_pattern
            shape = phase.shape
            assert Nx == shape[-2]
            assert Ny == shape[-1]
        # If it is a string
        elif isinstance(phase_pattern, str):
            if phase_pattern == 'lens':
                logger.debug("Phase pattern is a lens")
                #Get the focal length and wavelength from the kwargs
                focal_length = torch.tensor(self.kwargs['focal_length'])
                wavelength = torch.tensor(self.kwargs['wavelength'])
                #Create the lens phase
                phase = lensPhase(plane, wavelength, focal_length)
        else:
            logger.warning("Unsupported phase pattern : {}".format(phase_pattern))
            logger.warning("Setting uniform phase pattern")
            phase = self.uniform_phase(plane)
            raise Exception('unsupportedInitialization')
        return phase.to(torch.float64) # type: ignore

    def custom_amplitude(self, plane:plane.Plane, amplitude_pattern:torch.Tensor) -> torch.Tensor:
        Nx, Ny = plane.Nx, plane.Ny
        amplitude = amplitude_pattern
        shape = amplitude_pattern.shape
        assert Nx == shape[-2]
        assert Ny == shape[-1]
        return amplitude.to(torch.float64)

class Modulator(pl.LightningModule):
    def __init__(self, plane:plane.Plane, amplitude:torch.Tensor, phase:torch.Tensor):
        super().__init__()
        self.plane = plane
        self.amplitude = torch.nn.Parameter(amplitude, amplitude.requires_grad)
        self.phase = torch.nn.Parameter(phase, phase.requires_grad)

    def forward(self, input_wavefront = None) -> torch.Tensor:
        transmissivity = self.amplitude * torch.exp(1j *self.phase)
        if input_wavefront is None:
            input_wavefront = torch.ones_like(self.amplitude)

        return input_wavefront * transmissivity
    
    def print_info(self):
        self.plane.print_info()
        logger.info("Amplitude shape : {}".format(self.amplitude.shape))
        logger.info("Phase shape : {}".format(self.phase.shape))

    #--------------------------------
    # Setters
    #--------------------------------

    def set_amplitude(self, amplitude:torch.Tensor) -> None:
        self.amplitude = torch.nn.Parameter(amplitude)
        self.transmissivity = self.amplitude * torch.exp(1j * 2 * torch.pi * self.phase)

    def set_phase(self, phase:torch.Tensor) -> None:
        self.phase = torch.nn.Parameter(phase)
        self.transmissivity = self.amplitude * torch.exp(1j * 2 * torch.pi * self.phase)

    def set_transmissivity(self, transmissivity:torch.Tensor) -> None:
        self.transmissivity = transmissivity
        self.amplitude = torch.abs(transmissivity)
        self.phase = torch.angle(transmissivity)

    #--------------------------------
    # Getters
    #--------------------------------

    def get_amplitude(self, with_grad:bool=True) -> torch.Tensor:
        if with_grad:
            return self.amplitude
        else:
            return self.amplitude.detach()

    def get_phase(self, with_grad:bool=True) -> torch.Tensor:
        if with_grad:
            return self.phase
        else:
            return self.phase.detach()

    def get_transmissivity(self, with_grad:bool=True) -> torch.Tensor:
        if with_grad:
            return self.transmissivity
        else:
            return self.transmissivity.detach()

#--------------------------------
# Custom modulator functions
#--------------------------------

def lensPhase(lens_plane:plane.Plane, wavelength:torch.Tensor, focal_length:torch.Tensor) -> torch.Tensor:
    xx,yy = lens_plane.xx, lens_plane.yy
    Nx, Ny = lens_plane.Nx, lens_plane.Ny
    phase = -(xx**2 + yy**2) / ( 2 * focal_length )
    phase *= (2 * torch.pi / wavelength)
    phase = phase.view(1,1,Nx,Ny)
    return phase

def spherical_phase(lens_plane: plane.Plane, radius_of_curvature: torch.Tensor) -> torch.Tensor:
    xx, yy = lens_plane.xx, lens_plane.yy
    phase = (xx**2 + yy**2) / (2 * radius_of_curvature)
    return phase

def fresnel_phase(lens_plane: plane.Plane, focal_length: torch.Tensor, wavelength: torch.Tensor) -> torch.Tensor:
    xx, yy = lens_plane.xx, lens_plane.yy
    r_squared = xx**2 + yy**2
    phase = torch.sqrt(focal_length**2 + r_squared) - focal_length
    phase *= (2 * torch.pi / wavelength)
    return phase

def grating_phase(lens_plane: plane.Plane, grating_period: torch.Tensor) -> torch.Tensor:
    xx = lens_plane.xx
    phase = (2 * torch.pi / grating_period) * xx
    return phase

def vortex_phase(lens_plane: plane.Plane, charge: int) -> torch.Tensor:
    xx, yy = lens_plane.xx, lens_plane.yy
    phase = charge * torch.atan2(yy, xx)
    return phase

def zernike_polynomial_phase(lens_plane: plane.Plane, zernike_polynomial: torch.Tensor) -> torch.Tensor:
    xx, yy = lens_plane.xx, lens_plane.yy
    phase = zernike_polynomial(xx, yy)
    return phase

def zernike_polynomial(N: int, M: int, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
    rho = torch.sqrt(xx**2 + yy**2)
    theta = torch.atan2(yy, xx)
    radial = radial_zernike(N, M, rho)
    azimuthal = azimuthal_zernike(M, theta)
    return radial * azimuthal

def radial_zernike(N: int, M: int, rho: torch.Tensor) -> torch.Tensor:
    radial = torch.zeros_like(rho)
    for k in range((N-M)//2 + 1):
        radial += (-1)**k * torch.pow(rho, N-2*k) / (torch.pow(2, N) * torch.factorial(k) * torch.factorial((N+M)/2 - k) * torch.factorial((N-M)/2 - k))
    return radial

def azimuthal_zernike(M: int, theta: torch.Tensor) -> torch.Tensor:
    azimuthal = torch.zeros_like(theta)
    for k in range((M+1)//2):
        azimuthal += (-1)**k * torch.cos(theta) * torch.pow(torch.sin(theta), M-2*k) * torch.comb(M, k)
    return azimuthal


#--------------------------------
# Initialize: Test code
#--------------------------------
if __name__ == "__main__":
    import plane

    Nx = 1920
    Ny = 1080
    wavelength = torch.tensor(1.55e-6)
    focal_length = torch.tensor(10.e-2)
    
    plane_params = { 
                "name" : "test_plane",
                "center" : (0,0,0),
                "size" : (8.96e-3, 8.96e-3),
                "Nx" : Nx,
                "Ny" : Ny,
                "normal" : (0,0,1),
            }

    #Need to create the geometry for the modulators
    test_plane = plane.Plane(plane_params)
    lens_params = {
                "amplitude_init" : 'uniform',
                "phase_init" : 'custom',
                "gradients" : 'none',
                "phase_pattern" : 'lens',
                "amplitude_pattern" : None,
                "kwargs":{
                    "focal_length" : focal_length,
                    "wavelength" : wavelength,
                    },
            }
 
    mod_factory = ModulatorFactory()

    #You can call the ModulatorFactory directly
    lens = mod_factory(test_plane, lens_params)

    #Or you can directly call the creation function
    #lens = mod_factory.create_modulator(test_plane, lens_params)

    lens.print_info()

    random_params = {
                "amplitude_init" : 'random',
                "phase_init" : 'random',
                "gradients" : 'none',
                "phase_pattern" : None,
                "amplitude_pattern" : None,
                "kwargs":{},
            }

    mod = mod_factory(test_plane, random_params)
    mod.print_info()

