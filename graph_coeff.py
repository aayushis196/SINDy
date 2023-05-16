import torch
import matplotlib.pyplot as plt
import numpy as np

# Set the dimensions of the array
n_rows = 16
n_cols = 169

# Create an array of ones with shape (n_rows, n_cols)
arr_ones = np.ones((n_rows, n_cols))

# Multiply the array of ones by a sequence of integers starting from 0
arr_x = np.multiply(arr_ones, np.arange(n_rows)[:, np.newaxis])
arr_y = np.tile(np.arange(16), (169, 1)).T


#print(arr_x.shape)
#print(arr_y.shape)
# Define the tensor function
def tensor(data):
    return torch.tensor(data)

# Read the tensor from the file
with open('sindy_coefficients_polyorder1.txt', 'r') as f:
    tensor_str = f.read()
tensor_tmp = eval(tensor_str)

# Convert the tensor string to tensor
tensor_data = tensor_tmp.numpy()
tensor_data = tensor_data.transpose()
#plt.scatter(arr_x,arr_y,s=10,c=tensor_data, cmap='viridis')
print(tensor_data.shape[0])
#plt.style.use('dark_background')




x, y = np.meshgrid(np.arange(0, tensor_data.shape[1]), np.arange(0, tensor_data.shape[0]))

"""
plt.figure(figsize=(10, 8))
plt1 = plt.scatter(x, y, s=7, c=tensor_data,cmap='jet')
plt.colorbar(plt1, orientation='horizontal')
"""


"""
thresh = 0.01
# Create figure and axis objects
fig, axs = plt.subplots(figsize=(8, 4))

# Set white background
axs.set_facecolor('white')

# Create a boolean mask for the values in tensor_data between -0.01 and 0.01
mask = (tensor_data < -0.01) | (tensor_data > 0.01)

# Plot the scatter plot with the filtered data
plt1 = plt.scatter(x[mask], y[mask], s=7, c=tensor_data[mask], cmap='jet')
plt.colorbar(plt1,ax=axs, orientation='horizontal')

# Remove the grid
axs.grid(False)

plt.show()

"""
x = np.linspace(0, 1, tensor_data.shape[1])
y = np.linspace(0, 1, tensor_data.shape[0])
X, Y = np.meshgrid(x, y)

print(x.shape)
print(y.shape)
# Set up figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor('white')

# Define the threshold values
threshold = 1e-3

# Create a masked array where values outside the threshold are masked
masked_tensor_data = (tensor_data < -threshold) | (tensor_data > threshold)
masked_tensor_data_chat = np.ma.masked_outside(tensor_data, -threshold, threshold)

print(masked_tensor_data.shape)
# Create the scatter plot
scatter = ax.scatter(X[masked_tensor_data], Y[masked_tensor_data], s=7, c=tensor_data[masked_tensor_data], cmap='jet')

# Add vertical gray background behind each column
width = 0.002
# for i in range(0, 169):
#     if(np.all(masked_tensor_data_chat[:,i])):
#         ax.axvspan(x[i]-width, x[i]+width, facecolor='gray', alpha=0.1)

# Set the axis labels
#ax.set_xlabel('Polynomial latent coefficient')
#ax.set_ylabel('Y')
plt.tick_params(axis='both', which='both', length=0)

plt.grid(True)
# Add a colorbar
cbar = plt.colorbar(scatter,ax=ax, orientation='horizontal')
cbar.ax.set_title('Coeffecient Value')

# Show the plot
plt.show()
