
# The goal for this script is to resample an image to a new size and spatial resolution.
# This is important when trying to compare an ideal object to a simulated image.
# The simulated image will be from a sensor that is smaller and that has a different pixel size.

import torch
from plane.plane import Plane

def crop_or_pad(obj, plane1):
    # Assumes we get an object that has the same discretization as plane1
    # Sensor is smaller than the object
    _, _, obj_nx, obj_ny = obj.shape
    plane_shape = (1, 1, plane1.Nx, plane1.Ny)
    
    # If the object has more pixels than the sensor, crop the object
    if obj_nx > plane1.Nx or obj_ny > plane1.Ny:
        obj = center_crop(plane1, obj)

    # If the object has fewer pixels than the sensor, pad the object
    elif obj_nx < plane1.Nx or obj_ny < plane1.Ny:
        pad_x = int(torch.ceil((plane1.Nx - obj_nx) / 2))
        pad_y = int(torch.ceil((plane1.Ny - obj_ny) / 2))
        # Pad the object with zeros
        obj = torch.nn.functional.pad(obj, (pad_y, pad_y, pad_x, pad_x), mode='constant', value=0)
        obj = center_crop(plane1, obj)
    else:
        pass

    return obj

def center_crop(plane0, obj):
    # Center crop the object to the size of the sensor (plane0)
    _, _, obj_w, obj_h = obj.shape  # Get the width and height of the object

    crop_w = plane0.Nx
    crop_h = plane0.Ny

    # Compute the starting index for the crop
    start_w = obj_w // 2 - crop_w // 2
    start_h = obj_h // 2 - crop_h // 2

    # Perform the crop on the last two dimensions
    obj_cropped = obj[:, :, start_w:start_w + crop_w, start_h:start_h + crop_h]

    return obj_cropped


def resample(plane0, obj, plane1):
    # Resample the object (defined with the resolution of plane0) to the resolution of plane1
    nx = int(plane0.Lx / plane1.delta_x)
    ny = int(plane0.Ly / plane1.delta_y)
    # Interpolate the object to the new spatial resolution using torch.nn.functional.interpolate
    obj = torch.nn.functional.interpolate(obj, size=(nx, ny), mode='bilinear', align_corners=True)
    return obj



def create_cross_pattern(plane, cross_size):
    # Create a cross pattern on the plane
    xx, yy = plane.xx, plane.yy
    cross = torch.zeros_like(xx)
    cross[(torch.abs(xx) < cross_size) | (torch.abs(yy) < cross_size)] = 1.0

    return cross.view(1,1,plane.Nx,plane.Ny)


def spatial_resample(plane0, obj, plane1):
    obj_resampled = resample(plane0, obj, plane1)
    obj_sensor = crop_or_pad(obj_resampled, plane1)
    return obj_sensor


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Object plane
    params_plane0 = {
        'name': 'plane0',
        'center': [0,0,0],
        'size': [15.36, 8.64],
        'normal': [0,0,1],
        'Nx': 1920,
        'Ny': 1080,
        }

    # Smaller sensor plane
    params_plane1 = {
        'name': 'plane1',
        'center': [0,0,0],
        'size': [9.6768, 5.4432],
        'normal': [0,0,1],
        'Nx': 1920,
        'Ny': 1080,
        }

    plane0 = Plane(params_plane0)
    plane1 = Plane(params_plane1)


    cross_size = 0.2
    # Create an object that is larger than the sensor
    obj0_original = create_cross_pattern(plane0, cross_size)
    obj0_sensor = spatial_resample(plane0, obj0_original, plane1)

    obj1_original = create_cross_pattern(plane1, cross_size)
    obj1_sensor = spatial_resample(plane1, obj1_original, plane0)

    # Plot the results with pcolormesh
    fig, axs = plt.subplots(2, 2, figsize=(10,10))

    axs[0,0].pcolormesh(plane0.xx, plane0.yy, obj0_original.squeeze().numpy())
    axs[0,0].set_title(f'Object Plane (Original)\n Nx,Ny={obj0_original.shape[-2], obj0_original.shape[-1]}')

    axs[0,1].pcolormesh(plane1.xx, plane1.yy, obj0_sensor.squeeze().numpy())
    axs[0,1].set_title(f'Sensor Plane (Resampled)\n Nx,Ny={obj0_sensor.shape[-2], obj0_sensor.shape[-1]}')

    axs[1,0].pcolormesh(plane1.xx, plane1.yy, obj1_original.squeeze().numpy())
    axs[1,0].set_title(f'Object Plane (Original)\n Nx,Ny={obj1_original.shape[-2], obj1_original.shape[-1]}')

    axs[1,1].pcolormesh(plane0.xx, plane0.yy, obj1_sensor.squeeze().numpy())
    axs[1,1].set_title(f'Sensor Plane (Resampled)\n Nx,Ny={obj1_sensor.shape[-2], obj1_sensor.shape[-1]}')


    for ax in axs.flatten():
        ax.set_aspect('equal')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')

    plt.tight_layout()
    plt.show()


