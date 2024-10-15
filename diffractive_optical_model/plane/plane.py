import torch
from loguru import logger
import numpy as np

class Plane():
    def __init__(self, params:dict, bits:int=64)->None:
        self. params = params
        self.name = params['name']
        logger.debug("Initializing plane {}".format(self.name))

        self.center_x, self.center_y, self.center_z = torch.tensor(params['center'])
        self.Lx, self.Ly = torch.tensor(params['size'])
        self.Nx = torch.tensor(params['Nx'])
        self.Ny = torch.tensor(params['Ny'])

        
        # Fix types
        self.bits = bits
        self.fix_types(bits=self.bits)

        self.center = torch.tensor([self.center_x, self.center_y, self.center_z])
        self.size = torch.tensor([self.Lx, self.Ly])
        self.normal = torch.tensor(params['normal']).float()

        # Normalize the input and output normal vectors.
        self.normal = self.normal / torch.norm(self.normal)

        self.rot = self.create_rotation_matrix(self.normal, torch.tensor([0,0,1]))

        self.build_plane()

    def fix_types(self, bits=64):
        logger.debug("Fixing types for plane {}".format(self.name))
        if bits == 128:
            self.center_x = self.center_x.to(torch.float64)
            self.center_y = self.center_y.to(torch.float64)
            self.center_z = self.center_z.to(torch.float64)
            self.Lx = self.Lx.to(torch.float64)
            self.Ly = self.Ly.to(torch.float64)
            self.Nx = self.Nx.to(torch.int64)
            self.Ny = self.Ny.to(torch.int64)
            self.complex_type_torch = torch.complex128
            self.complex_type_numpy = np.complex128
            self.real_type_torch = torch.float64
            self.real_type_numpy = np.float64
        elif bits == 64:
            self.center_x = self.center_x.to(torch.float64)
            self.center_y = self.center_y.to(torch.float64)
            self.center_z = self.center_z.to(torch.float64)
            self.Lx = self.Lx.to(torch.float64)
            self.Ly = self.Ly.to(torch.float64)
            self.Nx = self.Nx.to(torch.int64)
            self.Ny = self.Ny.to(torch.int64)
            self.complex_type_torch = torch.complex64
            self.complex_type_numpy = np.complex64
            self.real_type_torch = torch.float64
            self.real_type_numpy = np.float64
        else:
            logger.error("Invalid number of bits.")
            raise ValueError("Invalid number of bits.")

    def build_plane(self)->None:
        logger.debug("Building plane {}".format(self.name))
        x = torch.div(self.Lx, 2)
        y = torch.div(self.Ly, 2)
        self.x = torch.linspace(-x, x, self.Nx, dtype=self.real_type_torch)
        self.y = torch.linspace(-y, y, self.Ny, dtype=self.real_type_torch)
        
        self.delta_x = torch.diff(self.x)[0]
        self.delta_y = torch.diff(self.y)[0]

        self.xx,self.yy = torch.meshgrid(self.x, self.y, indexing='ij')

        # Added these to help with DNI propagation.
        self.x_padded = torch.linspace(-self.Lx, self.Lx, 2*int(self.Nx), dtype=self.real_type_torch)
        self.y_padded = torch.linspace(-self.Ly, self.Ly, 2*int(self.Ny), dtype=self.real_type_torch)
        self.xx_padded,self.yy_padded = torch.meshgrid(self.x_padded, self.y_padded, indexing='ij')

        # FFT frequencies
        # Added these to assist with CZT propagation.
        # Need to convert the numpy initializations to a tensor to keep 128 bit precision.
        self.fx = torch.tensor(np.fft.fftfreq(int(self.Nx), d=self.delta_x.numpy()), dtype=self.real_type_torch)
        self.fy = torch.tensor(np.fft.fftfreq(int(self.Ny), d=self.delta_y.numpy()), dtype=self.real_type_torch)
        self.fxx,self.fyy = torch.meshgrid(self.fx, self.fy, indexing='ij')

        self.delta_fx = torch.diff(self.fx)[0]
        self.delta_fy = torch.diff(self.fy)[0]

        self.fx_padded = torch.tensor(np.fft.fftfreq(2*int(self.Nx), d=self.delta_x.numpy()), dtype=self.real_type_torch)
        self.fy_padded = torch.tensor(np.fft.fftfreq(2*int(self.Ny), d=self.delta_y.numpy()), dtype=self.real_type_torch)
        self.fxx_padded,self.fyy_padded = torch.meshgrid(self.fx_padded, self.fy_padded, indexing='ij')

        self.delta_fx_padded = torch.diff(self.fx_padded)[0]
        self.delta_fy_padded = torch.diff(self.fy_padded)[0]

    def print_info(self):
        logger.info("Plane {}:".format(self.name))
        logger.info("Center: {}".format(self.center))
        logger.info("Size: {}".format(self.size))
        logger.info("Samples: {}".format((self.Nx, self.Ny)))
        logger.info("Normal vector: {}".format(self.normal))
        logger.info("Rotation matrix: {}".format(self.rot))

        logger.info("x dtype: {}".format(self.x.dtype))
        logger.info("y dtype: {}".format(self.y.dtype))
        logger.info("xx dtype: {}".format(self.xx.dtype))
        logger.info("yy dtype: {}".format(self.yy.dtype))
        logger.info("fx dtype: {}".format(self.fx.dtype))
        logger.info("fy dtype: {}".format(self.fy.dtype))
        logger.info("fxx dtype: {}".format(self.fxx.dtype))
        logger.info("fyy dtype: {}".format(self.fyy.dtype))
        logger.info("x_padded dtype: {}".format(self.x_padded.dtype))
        logger.info("y_padded dtype: {}".format(self.y_padded.dtype))
        logger.info("xx_padded dtype: {}".format(self.xx_padded.dtype))
        logger.info("yy_padded dtype: {}".format(self.yy_padded.dtype))
        logger.info("fx_padded dtype: {}".format(self.fx_padded.dtype))
        logger.info("fy_padded dtype: {}".format(self.fy_padded.dtype))
        logger.info("fxx_padded dtype: {}".format(self.fxx_padded.dtype))
        logger.info("fyy_padded dtype: {}".format(self.fyy_padded.dtype))
        

    def plot2d(self, ax):
        logger.debug("Plotting plane {}".format(self.name))
        ###
        # Given an axis, plot the plane in the x-z plane.
        ###
    
        if self.rot is None:
            logger.debug("Rotation matrix is None.")
            self.rot = torch.eye(3)
        top_left_point = self.rot @ torch.tensor([-self.Lx/2, -self.Ly/2, 0]) + torch.tensor([self.center_x, self.center_y, self.center_z]) 
        bottom_right_point = self.rot @ torch.tensor([self.Lx/2, self.Ly/2, 0]) + torch.tensor([self.center_x, self.center_y, self.center_z])

        ax.plot([top_left_point[0], bottom_right_point[0]], [top_left_point[2], bottom_right_point[2]], 'k-')

        # Plot the normal vector
        ax.quiver(self.center_x, self.center_z, self.normal[0], self.normal[2], color='r', angles='xy', scale_units='xy', scale=1)

        return ax

    def create_rotation_matrix(self, input_normal, output_normal):
        logger.debug("Creating rotation matrix")
        # This function creates a rotation matrix to rotate the input and output
        # normal vectors to each other.
        input_normal = input_normal.float()
        output_normal = output_normal.float()

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

        #Check that the rotation matrix is valid and remove nans.
        if torch.isnan(rot_matrix).any():
            logger.error("Rotation matrix contains nans.")
            logger.error("Rotation matrix: {}".format(rot_matrix))
            raise ValueError("Rotation matrix contains nans.")

        return rot_matrix


    def are_antiparallel(self, vec1, vec2, tolerance=1e-6):
        # Normalize the vectors
        vec1_normalized = vec1 / torch.norm(vec1)
        vec2_normalized = vec2 / torch.norm(vec2)

        # Calculate the dot product
        dot_product = torch.dot(vec1_normalized, vec2_normalized)

        # Check if the dot product is close to -1
        return torch.isclose(dot_product, torch.tensor(-1.0), atol=tolerance)

    def is_same_spatial(self, plane):
        checks = []

        checks.append(torch.isclose(self.Lx, plane.Lx))
        checks.append(torch.isclose(self.Ly, plane.Ly))
        checks.append(torch.isclose(self.Nx, plane.Nx))
        checks.append(torch.isclose(self.Ny, plane.Ny))
        checks.append(torch.isclose(self.delta_x, plane.delta_x))
        checks.append(torch.isclose(self.delta_y, plane.delta_y))
        checks = [not(check) for check in checks]

        return not(any(checks))

    def is_smaller(self, plane):
        checks = []
        checks.append(self.Lx < plane.Lx)
        checks.append(self.Ly < plane.Ly)
        return all(checks)

    def scale(self, scale_factor, inplace=False):
        if inplace:
            self.Lx = self.Lx * scale_factor
            self.Ly = self.Ly * scale_factor
            self.build_plane()
        else:
            new_plane = Plane(self.params, bits=self.bits)
            new_plane.Lx = self.Lx * scale_factor
            new_plane.Ly = self.Ly * scale_factor
            new_plane.build_plane()
            return new_plane

    def __repr__(self):
        return "Plane: {}".format(self.name)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import yaml
    config = yaml.safe_load(open("../config.yaml"))
    plane_params = config['planes'][0]
    print(plane_params)


    plane = Plane(plane_params, bits=64)
    plane2 = Plane(plane_params, bits=128)

    from IPython import embed; embed()


