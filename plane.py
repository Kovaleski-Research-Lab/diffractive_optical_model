from os import name
import torch
from loguru import logger

class Plane():
    def __init__(self, params:dict)->None:
        self.name = params['name']
        logger.debug("Initializing plane {}".format(self.name))
        self.center_x, self.center_y, self.center_z = params['center']
        self.Lx, self.Ly = params['size']
        self.Nx = params['Nx']
        self.Ny = params['Ny']
        self.center = torch.tensor([self.center_x, self.center_y, self.center_z])
        self.size = torch.tensor([self.Lx, self.Ly])
        self.normal = torch.tensor(params['normal']).float()

        # Normalize the input and output normal vectors.
        self.normal = self.normal / torch.norm(self.normal)

        self.rot = self.create_rotation_matrix(self.normal, torch.tensor([0,0,1]))

        self.build_plane()

    def build_plane(self)->None:
        logger.debug("Building plane {}".format(self.name))
        x = torch.div(self.Lx, 2)
        y = torch.div(self.Ly, 2)
        self.x = torch.linspace(-x, x, self.Nx)
        self.y = torch.linspace(-y, y, self.Ny)
        
        self.delta_x = torch.diff(self.x)[0]
        self.delta_y = torch.diff(self.y)[0]

        self.xx,self.yy = torch.meshgrid(self.x, self.y, indexing='ij')

    def print_info(self):
        logger.info("Plane {}:".format(self.name))
        logger.info("Center: {}".format(self.center))
        logger.info("Size: {}".format(self.size))
        logger.info("Samples: {}".format((self.Nx, self.Ny)))
        logger.info("Normal vector: {}".format(self.normal))
        logger.info("Rotation matrix: {}".format(self.rot))

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


if __name__ == "__main__":

    import matplotlib.pyplot as plt


