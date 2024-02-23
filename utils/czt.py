

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.signal import fftconvolve
from scipy.signal import czt

from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_asm_transfer_function(fxx,fyy,z,k,wavelength):
    
    #Mask out non-propagating waves
    mask = np.sqrt(fxx**2 + fyy**2) < (1/wavelength)
    fxx = mask * fxx
    fyy = mask * fyy
    
    #10.1364/JOSAA.401908 equation 28
    #Also Goodman eq 3-78
    fz = np.sqrt(1 - (wavelength*fxx)**2 - (wavelength*fyy)**2)
    fz *= ((np.pi * 2)/wavelength)
    
    # Get the distance between the input and output planes.
    distance = z
    
    H = np.exp(1j * distance * fz)
    
    # Normalize the transfer function
    mag = np.abs(H)
    ang = np.angle(H)
    mag = mag / np.max(mag)
    H = mag * np.exp(1j*ang)
    
    H = np.fft.fftshift(H)

    return H



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
    
    print("strict_distance = ", strict_distance)
    return(np.less_equal(distance, strict_distance))

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def fix_numericals(wavefront):
    mag = np.abs(wavefront)
    phase = np.angle(wavefront)
    mag = np.ones_like(mag)
    wavefront = mag * np.exp(1j*phase)
    return wavefront

if __name__ == "__main__":

    # Source plane parameters
    lx = 20.e-3
    ly = 20.e-3
    nx = 1080
    ny = 1080

    x = np.linspace(-lx/2, lx/2, nx)
    y = np.linspace(-ly/2, ly/2, ny)
    xx, yy = np.meshgrid(x, y)
    dx = np.diff(x)[0]
    dy = np.diff(y)[0]

    z = 9.6e-2
    wavelength = 1.55e-6

    k = 2*np.pi/wavelength

    # Destination plane parameters
    lx_d = 20.e-3
    ly_d = 20.e-3
    nx_d = 1080
    ny_d = 1080

    x_d = np.linspace(-lx_d/2, lx_d/2, nx_d)
    y_d = np.linspace(-ly_d/2, ly_d/2, ny_d)
    xx_d, yy_d = np.meshgrid(x_d, y_d)

    dx_d = np.round(lx_d / nx_d, 6)
    dy_d = np.round(ly_d / ny_d, 6)
    print("dx_d = ", dx_d)
    print("dy_d = ", dy_d)

    fx = np.fft.fftfreq(2*nx, lx/nx)
    fy = np.fft.fftfreq(2*ny, lx/nx)
    fxx, fyy = np.meshgrid(fx, fy)

    dfx = np.diff(fx)[0]
    dfy = np.diff(fy)[0]
    print("dfx = ", dfx)
    print("dfy = ", dfy)

    # Create the input wavefront - a point source
    input_wavefront = create_input_wavefront(xx,yy)

    if(not check_distance(nx, ny, dx, dy, z, wavelength)):
        # Create the RSC transfer function
        print("Using RSC")
        H = create_rsc_transfer_function(xx,yy,z,k,wavelength)
    else:
        # Create the ASM transfer function
        print("Using ASM")
        H = create_asm_transfer_function(fxx,fyy,z,k,wavelength)

    ####################
    # CZT section 5 of 10.1364/JOSAA.31.001832
    ####################

    # Scaling factors
    alpha_x = np.round(dx_d/dfx, 10)
    alpha_y = np.round(dy_d/dfy, 10)

    print("alpha_x = ", alpha_x)
    print("alpha_y = ", alpha_y)

    # New coordinates
    wx = alpha_x*fx
    wy = alpha_y*fy
    print("wx = ", wx)
    print("wy = ", wy)


    wxx, wyy = np.meshgrid(wx, wy)
    dwx = np.round(np.diff(wx)[0], 10)
    dwy = np.round(np.diff(wy)[0], 10)
    print("dwx = ", dwx)
    print("dwy = ", dwy)

    assert np.allclose(dwx,dx_d), "dx_d = {} and dwx = {}".format(dx_d,dwx)
    assert np.allclose(dwy,dy_d), "dy_d = {} and dwy = {}".format(dy_d,dwy)
    #assert np.allclose(dx / dx_d , (2*nx) * dx**2 / alpha_x)
    #assert np.allclose(dy / dy_d , (2*ny) * dy**2 / alpha_y)

    C = np.exp(1j * np.pi * ((xx_d**2)/(alpha_x) + (yy_d**2)/(alpha_y)))
    D = np.exp(-1j * np.pi * ((wxx**2)/(alpha_x) + (wyy**2)/(alpha_y)))
    D = np.fft.fftshift(D)
    E = np.exp(1j * np.pi * ((wxx**2)/(alpha_x) + (wyy**2)/(alpha_y)))
    E = np.fft.fftshift(E)

    #C = fix_numericals(C)
    #D = fix_numericals(D)
    #E = fix_numericals(E)



    #fig,ax = plt.subplots(1,3,figsize=(15,5))
    #im0 = ax[0].imshow(np.angle(C), cmap='hsv')
    #ax[0].set_title('C')
    #divider = make_axes_locatable(ax[0])
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    #fig.colorbar(im0, cax=cax, orientation='vertical')    

    #im1 = ax[1].imshow(np.angle(D), cmap='hsv')
    #ax[1].set_title('D')
    #divider = make_axes_locatable(ax[1])
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    #fig.colorbar(im1, cax=cax, orientation='vertical')

    #im2 = ax[2].imshow(np.angle(E), cmap='hsv')
    #ax[2].set_title('E')
    #divider = make_axes_locatable(ax[2])
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    #fig.colorbar(im2, cax=cax, orientation='vertical')

    #plt.show()

    # Creating U^z of equation 35
    A = np.fft.fft2(input_wavefront)
    H = create_rsc_transfer_function(xx,yy,z,k,wavelength)
    A = np.fft.fftshift(A)
    H = np.fft.fftshift(H)
    Uz = A * H
    Uz = crop_center(Uz, nx, ny)

    common_x = Uz.shape[0] + D.shape[0] - 1
    common_y = Uz.shape[1] + D.shape[1] - 1

    d_padx = (common_x - D.shape[0])//2
    d_pady = (common_y - D.shape[1])//2

    u_padx = (common_x - Uz.shape[0])//2
    u_pady = (common_y - Uz.shape[1])//2

    print("common_x = ", common_x)
    print("common_y = ", common_y)
    print("d_padx = ", d_padx)
    print("d_pady = ", d_pady)
    print("u_padx = ", u_padx)
    print("u_pady = ", u_pady)


    Uz = np.pad(Uz, [(u_padx,u_padx), (u_pady,u_pady)], mode='constant')
    E = np.pad(E, [(d_padx,d_padx), (d_pady,d_pady)], mode='constant')
    D = np.pad(D, [(d_padx,d_padx), (d_padx,d_pady)], mode='symmetric')

    # Scale Uz - they call it U^z_w in the paper
    Uzw = Uz * E / (alpha_x*alpha_y)

    #Uzw = np.pad(Uzw, [(u_padx,u_padx), (u_padx,u_pady)], mode='constant')

    # Linear convolution of Uzw with D
    R = np.fft.fft2(Uzw)
    S = np.fft.fft2(D)
    Uzw_d = np.fft.ifft2(R * S)
    Uzw_d = np.fft.fftshift(Uzw_d)

    print("Uzw_d.shape = ", Uzw_d.shape)
    input()

    # Crop the result
    Uzw_d = np.pad(Uzw_d, [(1,1), (1,1)], mode='constant')
    
    Uzw_d = crop_center(Uzw_d, nx_d, ny_d)

    ## Scale the result
    uz = Uzw_d * C * dwx * dwy

    ## Normalize
    uz = normalize(uz)

    # RSC for comparison
    rsc_output = perform_rsc(input_wavefront,z,k,wavelength)
    # Crop
    rsc_output = crop_center(rsc_output, nx, ny)
    # Normalize
    rsc_output = normalize(rsc_output)


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

    im2 = ax[2].imshow(np.abs(np.abs(uz)-np.abs(rsc_output)), cmap='jet', vmin=0, vmax=1.)
    ax[2].set_title('Difference')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')


    plt.show()
