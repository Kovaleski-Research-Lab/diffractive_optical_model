#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
import os
import torch
import logging
from IPython import embed
import pytorch_lightning as pl

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
        self.modulator_type = self.params['modulator_type']
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
        self.amplitude = torch.nn.Parameter(amplitude)
        self.initialize_gradients()

    def set_phase(self, phase):
        phase = torch.tensor(phase).view(1,1,self.Nxm, self.Nym)
        self.phase = torch.nn.Parameter(phase)
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
        if self.modulator_type =='complex':
            logging.debug("Modulator | Keeping all gradients")
            pass
        elif self.modulator_type == 'phase_only':
            logging.debug("Modulator | Setting amplitude.requires_grad to False")
            self.amplitude.requires_grad = False
        elif self.modulator_type == 'amplitude_only':
            logging.debug("Modulator | Setting phase.requires_grad to False")
            self.phase.requires_grad = False
        elif self.modulator_type == 'none':
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
            amplitude = torch.nn.Parameter(torch.ones(1,1,self.Nxm, self.Nym))
        elif self.amplitude_initialization == 'random':
            amplitude = torch.nn.Parameter(torch.rand(1,1,self.Nxm, self.Nym))
        else:
            amplitude = torch.nn.Parameter(torch.ones(1,1,self.Nxm, self.Nym))
        return amplitude

    #--------------------------------
    # Initialize: Phases
    #--------------------------------

    def init_phase(self) -> torch.nn.Parameter:
        phase = None
        if self.phase_initialization == "uniform":
            logging.debug("Modulator | setting phase initialization to torch.ones()")
            phase = torch.nn.Parameter(torch.ones(1,1,self.Nxm, self.Nym))
        elif self.phase_initialization == "random":
            logging.debug("Modulator | setting phase initialization to torch.rand()")
            phase = torch.nn.Parameter(torch.rand(1,1,self.Nxm, self.Nym))
        elif self.phase_initialization == "lens":
            self.focal_length = torch.tensor(self.params['focal_length'])
            phase = -(self.xx**2 + self.yy**2) / ( 2 * self.focal_length )
            phase *= (2 * torch.pi / self.wavelength)
            phase = phase.view(1,1,self.Nxm,self.Nym)
            phase = torch.nn.Parameter(phase)

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
        modulator_type = params['type']

        phase_init = params['phase_init']
        amplitude_init = params['amplitude_init']

        phase_pattern = params['phase_pattern']
        amplitude_pattern = params['amplitude_pattern']

        if phase_init == 'uniform':
            phase = self.uniform_phase(plane)
        elif phase_init == 'random':
            phase = self.random_phase(plane)
        elif phase_init == 'custom':
            phase = self.custom_phase(plane, phase_pattern)
        else:
            raise Exception
        
        if amplitude_init == 'uniform':
            amplitude = self.uniform_amplitude(plane)
        elif amplitude_init == 'random':
            amplitude = self.random_amplitude(plane)
        elif amplitude_init == 'custom':
            amplitude = self.custom_amplitude(plane, amplitude_pattern)
        else:
            raise Exception

        if modulator_type == 'phase_only':
            phase.requires_grad = True
            amplitude.requires_grad = False
        elif modulator_type == 'amplitude_only':
            phase.requires_grad = False
            amplitude.requires_grad = True
        elif modulator_type == 'complex':
            phase.requires_grad = True
            amplitude.requires_grad = True
        elif modulator_type == None:
            phase.requires_grad = False
            amplitude.requires_grad = False
        else:
            phase.requires_grad = False
            amplitude.requires_grad = False
            #Log a non-critical error here

        modulator = Modulator(amplitude.clone(), phase.clone())
        return modulator

    def random_phase(self, plane) -> torch.Tensor:
        phase = torch.tensor(0)
        return phase

    def random_amplitude(self, plane) -> torch.Tensor:

        amplitude = torch.tensor(0)
        return amplitude

    def uniform_phase(self, plane) -> torch.Tensor:
        phase = torch.tensor(0)
        return phase

    def uniform_amplitude(self, plane) -> torch.Tensor:
        amplitude = torch.tensor(0)
        return amplitude

    def custom_phase(self, plane, phase_pattern) -> torch.Tensor:
        phase = torch.tensor(0)
        return phase

    def custom_amplitude(self, plane, amplitude_pattern) -> torch.Tensor:
        amplitude = torch.tensor(0)
        return amplitude


class Modulator(pl.LightningModule):
    def __init__(self, amplitude:torch.Tensor, phase:torch.Tensor):
        super().__init__()
        self.amplitude = amplitude
        self.phase = phase
        self.transmissivity = amplitude * torch.exp(1j * phase)

    def forward(self, input_wavefront):
        return input_wavefront * self.transmissivity
#--------------------------------
# Initialize: Test code
#--------------------------------
if __name__ == "__main__":
    import plane

    plane_params = { 
                "name" : "test_plane",
                "center" : (0,0),
                "size" : (8.96e-3, 8.96e-3),
                "Nx" : 1080,
                "Ny" : 1080,
            }

    lens_params = {
                "type" : 'phase_only',
                "phase_init" : 'lens',
                "amp_init": 'uniform',
                "phase_pattern" : None,
                "amplitude_pattern" : None,
            }
 
    #Need to create the geometry for the modulators
    plane = plane.Plane(plane_params)

    mod_factory = ModulatorFactory()

    #You can call the ModulatorFactory directly
    lens = mod_factory(plane, lens_params)
    #Or you can directly call the creation function
    #lens = mod_factory.create_modulator(plane, lens_params)


