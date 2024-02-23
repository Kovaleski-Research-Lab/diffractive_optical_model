import numpy as np
from scipy.signal import ZoomFFT
t = np.linspace(0, 1, 1021)
x = np.cos(2*np.pi*15*t) + np.sin(2*np.pi*17*t)
f1, f2 = 5, 27
transform = ZoomFFT(len(x), [f1, f2], len(x), fs=1021)
X = transform(x)
f = np.linspace(f1, f2, len(x))
X_fft = np.fft.fft(x)
X_fft = np.fft.fftshift(X_fft)

X_fft = X_fft[len(x)//4:3*len(x)//4]

import matplotlib.pyplot as plt

fig,ax = plt.subplots(2,1)
ax[0].plot(f, 20*np.log10(np.abs(X)))
ax[1].plot(20*np.log10(np.abs(X_fft)))

plt.show()
