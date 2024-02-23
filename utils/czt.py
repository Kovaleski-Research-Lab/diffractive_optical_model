

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.signal import fftconvolve

from mpl_toolkits.axes_grid1 import make_axes_locatable



def create_rsc_transfer_function(xx,yy,z,k,wavelength):
    r = np.sqrt(xx**2 + yy**2 + z**2) 
    h_rsc = np.exp(np.sign(z) * 1j*k*r)/r
    h_rsc *= ((1/r) - (1j/k))
    h_rsc *= (1/(2*np.pi))*(z/r)
    H = np.fft.fft2(h_rsc)
    return H

def create_input_wavefront(xx,yy):
    wavefront = xx**2 + yy**2 < 0.1e-6
    wavefront = wavefront.astype(np.float64)
    return wavefront

def perform_rsc(input_wavefront,z,k,wavelength, czt=False):
    H = create_rsc_transfer_function(xx,yy,z,k,wavelength)
    if czt:
        pass
    else:
        output_wavefront = np.fft.ifft2(H*np.fft.fft2(input_wavefront))
        output_wavefront = np.fft.fftshift(output_wavefront)
    return output_wavefront

def normalize(wavefront):
    mag = np.abs(wavefront)
    phase = np.angle(wavefront)
    mag = mag - np.min(mag)
    mag = mag / np.max(mag)
    wavefront = mag * np.exp(1j*phase)
    return wavefront

def check_distance(Nx, Ny, delta_x, delta_y, distance, wavelength):
    
    distance_criteria_y = 2 * delta_y * ( Ny * delta_y) / wavelength
    distance_criteria_y *= np.sqrt(1 - (wavelength / (2 * Ny))**2)
    
    distance_criteria_x = 2 * delta_x * ( Nx * delta_x) / wavelength
    distance_criteria_x *= np.sqrt(1 - (wavelength / (2 * Nx))**2)
    
    strict_distance = np.minimum(distance_criteria_y, distance_criteria_x) 
    
    return(np.less_equal(distance, strict_distance))

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

if __name__ == "__main__":


    # Source plane parameters
    lx = 2.e-3
    ly = 2.e-3
    nx = 256
    ny = 256
    x = np.linspace(-lx, lx, 2*nx)
    y = np.linspace(-ly, ly, 2*ny)
    xx, yy = np.meshgrid(x, y)
    dx = np.diff(x)[0]
    dy = np.diff(y)[0]

    z = 10.e-2
    wavelength = 1.55e-6

    fx = np.fft.fftfreq(2*nx, np.diff(x)[0])
    fy = np.fft.fftfreq(2*ny, np.diff(y)[0])
    fxx, fyy = np.meshgrid(fx, fy)
    dfx = np.diff(fx)[0]
    dfy = np.diff(fy)[0]

    k = 2*np.pi/wavelength

    # Destination plane parameters
    lx_d = 3.e-3
    ly_d = 3.e-3
    nx_d = 384
    ny_d = 384

    x_d = np.linspace(-lx_d/2, lx_d/2, nx_d)
    y_d = np.linspace(-ly_d/2, ly_d/2, ny_d)
    xx_d, yy_d = np.meshgrid(x_d, y_d)
    dx_d = np.diff(x_d)[0]
    dy_d = np.diff(y_d)[0]

    # Create the input wavefront - a point source
    input_wavefront = create_input_wavefront(xx,yy)

    assert(not check_distance(nx, ny, dx, dy, z, wavelength))

    # Create the RSC transfer function
    H = create_rsc_transfer_function(xx,yy,z,k,wavelength)

    ####################
    # CZT section 5 of 10.1364/JOSAA.31.001832
    ####################

    # Scaling factors
    alpha_x = dx_d/dfx
    alpha_y = dy_d/dfy

    # New coordinates
    wx = alpha_x*fx
    wy = alpha_y*fy
    wxx, wyy = np.meshgrid(wx, wy)
    dwx = np.diff(wx)[0]
    dwy = np.diff(wy)[0]

    assert np.allclose(dwx,dx_d), "dx_d = {} and dwx = {}".format(dx_d,dwx)
    assert np.allclose(dwy,dy_d), "dy_d = {} and dwy = {}".format(dy_d,dwy)
    assert np.allclose(dx / dx_d , (2*nx) * dx**2 / alpha_x)
    assert np.allclose(dy / dy_d , (2*ny) * dy**2 / alpha_y)

    C = np.exp(1j * np.pi * ((xx_d**2)/(alpha_x) + (yy_d**2)/(alpha_y)))
    D = np.exp(-1j * np.pi * ((wxx**2)/(alpha_x) + (wyy**2)/(alpha_y)))
    E = np.exp(1j * np.pi * ((wxx**2)/(alpha_x) + (wyy**2)/(alpha_y)))

    # Creating U^z of equation 35
    A = np.fft.fft2(input_wavefront)
    H = create_rsc_transfer_function(xx,yy,z,k,wavelength)
    A = np.fft.fftshift(A)
    H = np.fft.fftshift(H)
    Uz = A * H

    # Scale Uz - they call it U^z_w in the paper
    Uzw = Uz * E / (alpha_x*alpha_y)

    # Linear convolution of Uzw with D
    Uzw_d = fftconvolve(Uzw, D, mode='full')
    #Uzw_d = convolve2d(Uzw, D, mode='full', boundary='wrap')


    # Crop the result
    Uzw_d = crop_center(Uzw_d, nx_d, ny_d)

    plt.imshow(np.abs(Uzw_d))
    plt.show()

    # Scale the result
    uz = Uzw_d * C * dwx * dwy

    # Normalize
    uz = normalize(uz)

    # RSC for comparison
    rsc_output = perform_rsc(input_wavefront,z,k,wavelength)
    # Normalize
    rsc_output = normalize(rsc_output)
    # Crop
    rsc_output = crop_center(rsc_output, nx_d, ny_d)


    fig, ax = plt.subplots(1,3,figsize=(15,5))
    im0 = ax[0].imshow(np.abs(uz), cmap='jet')
    ax[0].set_title('CZT')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')

    im1 = ax[1].imshow(np.abs(rsc_output), cmap='jet')
    ax[1].set_title('RSC')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    im2 = ax[2].imshow(np.abs(uz)-np.abs(rsc_output), cmap='jet')
    ax[2].set_title('Difference')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')


    plt.show()
