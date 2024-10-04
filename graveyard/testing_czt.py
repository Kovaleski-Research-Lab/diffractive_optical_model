import numpy as np
from scipy.signal import CZT

# Example parameters (you will need to adjust these)
n = 256  # Size of the signal (e.g., for a 256x256 image)
m = 512  # Number of output points (can be different from n)
w = np.exp(-2j * np.pi / m)  # Example ratio between points
a = 1 + 0j  # Starting point in the complex plane

# Initialize the CZT objects for rows and columns
czt_rows = CZT(n=n, m=m, w=w, a=a)
czt_columns = CZT(n=n, m=m, w=w, a=a)

# Define or load your image
image = np.random.rand(n, n)  # Example: 256x256 random image

# Apply the CZT across rows
czt_result_rows = np.apply_along_axis(czt_rows, axis=0, arr=image)

# Apply the CZT across columns of the result from rows
czt_result_2d = np.apply_along_axis(czt_columns, axis=1, arr=czt_result_rows)

# czt_result_2d is the 2D CZT of the original image
