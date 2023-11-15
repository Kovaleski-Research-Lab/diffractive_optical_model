from os import name
import torch
from loguru import logger

class Plane():
    def __init__(self, params:dict)->None:

        self.name = params['name']
        self.center_x, self.center_y = params['center']
        self.Lx, self.Ly = params['size']
        self.Nx = params['Nx']
        self.Ny = params['Ny']
        self.build_plane()

    def build_plane(self)->None:
        logger.debug("Building plane {}".format(self.name))
        x = torch.div(self.Lx, 2)
        y = torch.div(self.Ly, 2)
        self.x = torch.linspace(-x, x, self.Nx)
        self.y = torch.linspace(-y, y, self.Ny)
        self.xx,self.yy = torch.meshgrid(self.x, self.y, indexing='ij')

    def print_info(self):
        logger.info("Name : {}, Center x : {}, Center y : {}, Lx : {}, Ly : {}, Nx : {}, Ny : {}".format(self.name, 
                                                      self.center_x, 
                                                      self.center_y,
                                                      self.Lx, 
                                                      self.Ly,
                                                      self.Nx,
                                                      self.Ny))

if __name__ == "__main__":

    params = {"name" : "test_plane",
              "center" : (1,1),
              "size" : (5,5),
              "Nx" : 10,
              "Ny" : 10,
            }
    plane = Plane(params)
    plane.print_info()
