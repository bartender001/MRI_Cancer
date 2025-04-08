import scipy.io
import numpy as np

breast_file = []
# Class 1 - Sample 1
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample01/maps/class_1_1_1.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample01/maps/class_1_1_2.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample01/maps/class_1_1_3.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample01/maps/class_1_1_4.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample01/maps/class_1_1_5.mat')

# # Class 1 - Sample 2
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample02/maps/class_1_2_1.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample02/maps/class_1_2_2.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample02/maps/class_1_2_3.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample02/maps/class_1_2_4.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample02/maps/class_1_2_5.mat')

# # Class 2 - Sample 1
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample01/maps/class_2_1_1.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample01/maps/class_2_1_2.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample01/maps/class_2_1_3.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample01/maps/class_2_1_4.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample01/maps/class_2_1_5.mat')

# # Class 2 - Sample 2
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample02/maps/class_2_2_1.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample02/maps/class_2_2_2.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample02/maps/class_2_2_3.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample02/maps/class_2_2_4.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample02/maps/class_2_2_5.mat')

# # Class 2 - Sample 3
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample03/maps/class_2_3_1.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample03/maps/class_2_3_2.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample03/maps/class_2_3_3.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample03/maps/class_2_3_4.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample03/maps/class_2_3_5.mat')

# # Class 3 - Sample 1
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample01/maps/class_3_1_1.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample01/maps/class_3_1_2.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample01/maps/class_3_1_3.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample01/maps/class_3_1_4.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample01/maps/class_3_1_5.mat')

# # Class 3 - Sample 2
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample02/maps/class_3_2_1.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample02/maps/class_3_2_2.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample02/maps/class_3_2_3.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample02/maps/class_3_2_4.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample02/maps/class_3_2_5.mat')

# # Class 3 - Sample 3
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample03/maps/class_3_3_1.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample03/maps/class_3_3_2.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample03/maps/class_3_3_3.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample03/maps/class_3_3_4.mat')
# breast_file.append('/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample03/maps/class_3_3_5.mat')




# breast_sigma = []
# for file_path in breast_file:
#     data = scipy.io.loadmat(file_path)
#     sigma = data['sigma']
#     breast_sigma.append(sigma)

# breast_sigma = np.array(breast_sigma)
# print(breast_sigma.shape[0])

# breast_eps = []
# for file_path in breast_file:
#     data = scipy.io.loadmat(file_path)
#     eps = data['epsilon_r']
#     breast_eps.append(eps)

# breast_eps = np.array(breast_eps)
# print(breast_eps.shape)

# solution_sigma = []
# for filepath in breast_file:
#     data = scipy.io.loadmat(filepath)
#     sigma = data['solution.sigma']
#     solution_sigma.append(sigma)

# import scipy.io
# import numpy as np


# file_path = '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample01/class1sample1cplane1.mat'
# data = scipy.io.loadmat(file_path)

# # Access the 'solution' struct
# solution = data['solution']  # 'solution' is a 1x1 struct

# # Extract 'sigma' from the 'solution' struct
# sigma = solution['sigma'][0, 0]  


# # Convert 'sigma' to a NumPy array
# sigma_array = np.array(sigma)

# # Verify the shape of 'sigma'
# print("Shape of sigma:", sigma_array.shape)

# # Optional: Print some values for verification
# print("Sample values from sigma:", sigma_array[:5, :5])

# epsilon_r = solution['epsilon_r'][0, 0]  # Access 'epsilon_r' field

# # Convert 'epsilon_r' to a NumPy array
# epsilon_r_array = np.array(epsilon_r)

# # Verify the shape of 'epsilon_r'
# print("Shape of epsilon_r:", epsilon_r_array.shape)
# print("Sample values from epsilon_r:", epsilon_r_array[:5, :5])


# import scipy.io
# import numpy as np


# file_path = ['/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample01/class1sample1cplane1.mat'
# , '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample01/class1sample1cplane2.mat']

# data = scipy.io.loadmat(file_path)

# solution_epsilon_r = []
# solution_sigma = []
# breast_epsilon_r = []
# breast_sigma = []

# for i in len(file_path):
#   solution = file_path[i]['solution']
#   breast = file_path[i]['breast']

#   solution_epsilon_r.append(solution['epsilon_r'][0,0])
#   solution_sigma.append(solution['sigma'][0,0])
#   breast_epsilon_r.append(breast['epsilon_r'][0,0])
#   breast_sigma.append(breast['sigma'][0,0])

# solution_epsilon_r = np.array(solution_epsilon_r)
# solution_sigma = np.array(solution_sigma)
# breast_epsilon_r = np.array(breast_epsilon_r)
# breast_sigma = np.array(breast_sigma)

import scipy.io
import numpy as np

file_paths = [
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample01/class1sample1cplane1.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample01/class1sample1cplane2.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample01/class1sample1cplane3.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample01/class1sample1cplane4.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample01/class1sample1cplane5.mat',

    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample02/class1sample2cplane1.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample02/class1sample2cplane2.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample02/class1sample2cplane3.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample02/class1sample2cplane4.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class1/sample02/class1sample2cplane5.mat',

    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample01/class2sample1cplane1.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample01/class2sample1cplane2.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample01/class2sample1cplane3.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample01/class2sample1cplane4.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample01/class2sample1cplane5.mat',

    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample02/class2sample2cplane1.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample02/class2sample2cplane2.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample02/class2sample2cplane3.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample02/class2sample2cplane4.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample02/class2sample2cplane5.mat',

    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample03/class2sample3cplane1.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample03/class2sample3cplane2.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample03/class2sample3cplane3.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample03/class2sample3cplane4.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class2/sample03/class2sample3cplane5.mat',

    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample01/class3sample1cplane1.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample01/class3sample1cplane2.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample01/class3sample1cplane3.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample01/class3sample1cplane4.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample01/class3sample1cplane5.mat',

    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample02/class3sample2cplane1.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample02/class3sample2cplane2.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample02/class3sample2cplane3.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample02/class3sample2cplane4.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample02/class3sample2cplane5.mat',

    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample03/class3sample3cplane1.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample03/class3sample3cplane2.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample03/class3sample3cplane3.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample03/class3sample3cplane4.mat',
    '/Library/gurobi1103/Microwave-Imaging/A Quadratic Programming Approach to Microwave Imaging/breast_phantoms/class3/sample03/class3sample3cplane5.mat'
]

solution_epsilon_r = []
solution_sigma = []
breast_epsilon_r = []
breast_sigma = []

for file_path in file_paths:
    data = scipy.io.loadmat(file_path)  # Load each file
    solution = data['solution']
    breast = data['testbench']

    # Extract data from the loaded file
    solution_epsilon_r.append(solution['epsilon_r'][0, 0])
    solution_sigma.append(solution['sigma'][0, 0])
    breast_epsilon_r.append(breast['epsilon_r'][0, 0])
    breast_sigma.append(breast['sigma'][0, 0])

# Convert lists to numpy arrays
solution_epsilon_r = np.array(solution_epsilon_r)
solution_sigma = np.array(solution_sigma)
breast_epsilon_r = np.array(breast_epsilon_r)
breast_sigma = np.array(breast_sigma)

print(solution_epsilon_r.shape)
print(solution_sigma.shape)
print(breast_epsilon_r.shape)
print(breast_sigma.shape)

# Save the arrays as .npy files
np.save('solution_epsilon_r.npy', solution_epsilon_r)
np.save('solution_sigma.npy', solution_sigma)
np.save('breast_epsilon_r.npy', breast_epsilon_r)
np.save('breast_sigma.npy', breast_sigma)

print("Files saved as .npy format.")