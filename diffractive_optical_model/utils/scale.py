
# This tests scaling an object using a magnification value

import torch

def create_cross_pattern(plane, cross_size):
    # Create a cross pattern on the plane
    xx, yy = plane.xx, plane.yy
    cross = torch.zeros_like(xx)
    cross[(torch.abs(xx) < cross_size) | (torch.abs(yy) < cross_size)] = 1.0

    return cross.view(1,1,plane.Nx,plane.Ny)

if __name__ == "__main__":
    from plane.plane import Plane
    import matplotlib.pyplot as plt

    # Object plane
    params_plane0 = {
        'name': 'plane0',
        'center': [0,0,0],
        'size': [8.64, 8.64],
        'normal': [0,0,1],
        'Nx': 1000,
        'Ny': 1000,
        }

    scale = 0.6

    plane0 = Plane(params_plane0)
    obj = create_cross_pattern(plane0, 0.2)

    scaled_plane = plane0.scale(scale, inplace=False)


    # Plot the original and the scaled using pcolormesh
    fig, axs = plt.subplots(1,2, figsize=(10,5))

    axs[0].pcolormesh(plane0.xx, plane0.yy, obj.squeeze().numpy())
    axs[0].set_title("Original object")

    axs[1].pcolormesh(scaled_plane.xx, scaled_plane.yy, obj.squeeze().numpy())
    axs[1].set_title("Scaled object")

    for ax in axs.flatten():
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()



    

