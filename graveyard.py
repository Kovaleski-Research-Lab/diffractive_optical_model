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
        return ax




# This function creates a rotation matrix to rotate the input and output
        # normal vectors to each other.
        input_normal = input_normal.float()
        output_normal = output_normal.float()

        # Normalize the input and output normal vectors.
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
