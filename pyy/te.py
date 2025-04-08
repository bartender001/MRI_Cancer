

import scipy.io
import numpy as np

# Load the .mat file
file_path = '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample01/maps/class_1_1_1.mat'  # Replace with your .mat file's path
data = scipy.io.loadmat(file_path)

# Access the 'sigma' grid
sigma = data['sigma']  # Directly access the 'sigma' variable

# Convert to a NumPy array (if needed)
sigma_array = np.array(sigma)

# Verify the shape
print("Shape of sigma:", sigma_array.shape)

# # Optional: Print some sample values to verify
# print(sigma_array.shape)


