import numpy as np
import matplotlib.pyplot as plt

def generate_breast_phantom(size=128, num_tumors=1):
    phantom = np.random.uniform(2.5, 10, (size, size))  # Base tissue permittivity
    for _ in range(num_tumors):
        x, y = np.random.randint(0, size, 2)
        radius = np.random.randint(5, 15)
        for i in range(size):
            for j in range(size):
                if (i-x)**2 + (j-y)**2 <= radius**2:
                    phantom[i, j] = np.random.uniform(20, 67)  # Tumor permittivity
    return phantom

# Generate a phantom
phantom = generate_breast_phantom()
plt.imshow(phantom, cmap='viridis')
plt.colorbar()
plt.title("Simulated Breast Phantom with Tumor")
plt.show()