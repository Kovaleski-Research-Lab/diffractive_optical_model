from loguru import logger
import matplotlib.pyplot as plt

import sys
import torch

sys.path.append("../")
from plane import Plane



if __name__ == "__main__":

    # Create a list of normal vectors to test.
    normals_list = [torch.tensor([i,0,j]) for i in range(-2,2) for j in range(-2,2)]
    logger.info("Normals list: {}".format(normals_list))

    fig,axis = plt.subplots(1,1, figsize=(5,5))
    for i,normal in enumerate(normals_list):
        logger.warning("Normal vector: {}. Index: {}".format(normal, i)) 
        params = {"name" : "test_plane",
                  "center" : (0,0,i),
                  "size" : (1,1),
                  "normal" : normal,
                  "Nx" : 10,
                  "Ny" : 10,
                }

        plane = Plane(params)
        plane.print_info()
        plane.plot2d(axis)

    from IPython import embed; embed()

    axis.set_xlim(-1,len(normals_list)+1)
    axis.set_ylim(-1,len(normals_list)+1)

    plt.show()
    pass
