import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.core import datamodule
import yaml
import propagator
import modulator
import datamodule
import source
from IPython import embed; 

if __name__ == "__main__":

    params = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)['don']

    prop_params = params['propagators']
    mod_params = params['modulators']
    source_params = params['sources']
    num_propagators = len(prop_params)

    layers = []
    layers.append(source.Source(source_params[0]))
    for m,p in zip(mod_params, prop_params):
        layers.append(modulator.Modulator(mod_params[m]))
        layers.append(propagator.Propagator(prop_params[p]))

    for l in layers:
        print(l)

    u = None
    for i,l in enumerate(layers):
        if i == 0:
            u = l.forward()
        else:
            u = l.forward(u)


    image = u.abs().squeeze()**2
    plt.imshow(image)
    plt.show()
