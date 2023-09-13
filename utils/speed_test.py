import sys
import yaml
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
from IPython import embed
import time
sys.path.append("../")
from core import flir_8580, holoeye_pluto21, datamodule
from utils import parameter_manager

def create_flat(shape:tuple, value:int):
    return np.ones(shape, dtype="uint8") * value

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)
    #camera = flir_8580.FLIR_8580()
    #camera.set_buffer_handling_mode("newestonly")
    #camera.set_acquisition_mode("continuous")
    ##camera.set_nuc_source("internal")
    ##camera.perform_nuc()
    #camera.disable_header()

    #camera.begin_acquisition()

    logging.getLogger().setLevel(logging.ERROR)
    slm = holoeye_pluto21.HoloeyePluto()

    params = yaml.load(open('../config.yaml'), Loader = yaml.FullLoader)

    #Initialize the data module
    dm = datamodule.select_data(params)
    dm.prepare_data()
    dm.setup(stage="fit")

    dataloader = dm.train_dataloader()
    dl_iter = iter(dataloader)
    filename = "../slm_patterns/mnist_digit_0000.bmp"

    slm_file_path = "/root/mnist_digit_0000.bmp"

    slm_options = ['wl=1550']
    slm.create_display_command(filename=slm_file_path, options=slm_options)

    print(slm.command)

    #for i in range(0,255):
    #    print(i)
    #    slm_pattern = create_flat((1080,1920),i)
    #    pattern = Image.fromarray(slm_pattern)
    #    pattern.save(filename)
    #    slm.send_scp(filename)
    #    slm.update(filename=slm_file_path, options=slm_options)
    #    time.sleep(0.5)

    for i,batch in enumerate(tqdm(dl_iter)):
        slm_pattern = batch[1].squeeze().numpy()
        pattern = Image.fromarray(slm_pattern)
        pattern.save(filename)
        slm.send_scp(filename)
        slm.update(filename=slm_file_path, options=slm_options)
        #image = Image.fromarray(camera.get_image())
        #image.save('../results/image_{:04d}.png'.format(i))

    #    ##Send the pattern to the Holoeye
    #    #slm.send_scp(filename)
    #    ##Display the pattern on the slm
    #    #slm.update(filename=filename)
    #    ##Collect image 
    #    #image = camera.get_image()
