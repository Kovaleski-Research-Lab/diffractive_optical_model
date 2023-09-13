#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import torch
import logging
import pytorch_lightning as pl

#--------------------------------
# Initialize: Wavefront Modulator
#--------------------------------

class Modulator(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        logging.debug("modulator.py - Initializing Modulator")
        # Load : Copy of parameters 
        self.params = params.copy()

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


        # Create : The modulator
        self.create_modulator()
 
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

    def set_amplitude(self, amplitude):
        amplitude = torch.tensor(amplitude).view(1,1,self.Nxm, self.Nym)
        self.amplitude = torch.nn.Parameter(amplitude)
        self.initialize_gradients()

    #--------------------------------
    # Initialize: Amplitudes 
    #--------------------------------

    def init_amplitude(self) -> torch.nn.Parameter | None:
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

    def init_phase(self) -> torch.nn.Parameter | None:
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
  
        return amplitude,phase

    #--------------------------------
    # Initialize : Forward pass 
    #--------------------------------

    def forward(self, wavefront):
        if wavefront.squeeze().shape != self.amplitude.squeeze().shape : # type: ignore
            amplitude,phase = self.adjust_shape(wavefront)
        else:
            amplitude,phase = self.amplitude,self.phase

        layer = amplitude * torch.exp(1j*phase)
        return layer * wavefront

#--------------------------------
# Initialize: Lens Modulator
#--------------------------------

class Lens(Modulator):
    def __init__(self, params, focal_length):
        super().__init__(params)
        logging.debug("modulator.py - Initializing Lens")
        self.focal_length = focal_length
        logging.debug("Lens | Setting focal length to {}".format(self.focal_length))
        self.update_phases()

    def update_phases(self):
        #----------------------------------
        #          2pi     - ( x^2 + y^2 )
        # phase = ------ * ---------------
        #         lambda          2f
        #----------------------------------
        logging.debug("Lens | Updating phases to lens pattern")
        phase = -(self.xx**2 + self.yy**2) / ( 2 * self.focal_length )
        phase *= (2 * torch.pi / self.wavelength)
        phase = phase.view(1,1,self.Nxm,self.Nym)
        self.phase = torch.nn.Parameter(phase)

        self.initialize_gradients()
#--------------------------------
# Initialize: Test code
#--------------------------------

if __name__ == "__main__":
    import yaml
    import numpy as np
    import matplotlib.pyplot as plt
    logging.basicConfig(level=logging.DEBUG)
    params = yaml.load(open("../config.yaml"), Loader=yaml.FullLoader)
    params = params['don']['modulator'][0]

    modulator = Modulator(params)

    lens1 = Lens(params, focal_length = torch.tensor(0.0337))
    lens2 = Lens(params, focal_length = torch.tensor(0.60264/2))

    fig,ax = plt.subplots(1,3,figsize=(10,5))

    ax[0].imshow(torch.exp(1j*lens1.phase.squeeze()).angle().detach())
    ax[1].imshow(torch.exp(1j*lens2.phase.squeeze()).angle().detach())
    ax[2].imshow(torch.exp(1j*modulator.phase.squeeze().detach()).angle())

    plt.show()
    
