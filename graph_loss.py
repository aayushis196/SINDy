import torch
import matplotlib.pyplot as plt
import numpy as np

# Read the tensor from the file
with open('train_loss_polyorder1.txt', 'r') as f:
    tensor_str = f.read()
train_loss_1 = eval(tensor_str)

# Read the tensor from the file
with open('val_loss_polyorder1.txt', 'r') as f:
    tensor_str = f.read()
val_loss_1 = eval(tensor_str)

# Read the tensor from the file
with open('train_loss_polyorder2.txt', 'r') as f:
    tensor_str = f.read()
train_loss_2 = eval(tensor_str)

# Read the tensor from the file
with open('val_loss_polyorder2.txt', 'r') as f:
    tensor_str = f.read()
val_loss_2 = eval(tensor_str)

# Read the tensor from the file
with open('train_loss_polyorder3.txt', 'r') as f:
    tensor_str = f.read()
train_loss_3 = eval(tensor_str)

# Read the tensor from the file
with open('val_loss_polyorder3.txt', 'r') as f:
    tensor_str = f.read()
val_loss_3 = eval(tensor_str)

with open('train_loss_polyorder4.txt', 'r') as f:
    tensor_str = f.read()
train_loss_4 = eval(tensor_str)

# Read the tensor from the file
with open('val_loss_polyorder4.txt', 'r') as f:
    tensor_str = f.read()
val_loss_4 = eval(tensor_str)

x = np.linspace(0, 1, np.size(train_loss_2))

# Create a figure and plot the first graph
fig, ax = plt.subplots(figsize=(8, 5))

# ax.plot(x, train_loss_1, label='Polynomial Order=1')
# ax.legend(loc='upper right')

ax.plot(x, val_loss_1, label='Polynomial Order=1')
ax.legend(loc='upper right')

# ax.plot(x, train_loss_2, label='Polynomial Order=2')
# ax.legend(loc='upper right')

ax.plot(x, val_loss_2, label='Polynomial Order=2')
ax.legend(loc='upper right')

# ax.plot(x, train_loss_3, label='Polynomial Order=3')
# ax.legend(loc='upper right')

ax.plot(x, val_loss_3, label='Polynomial Order=3')
ax.legend(loc='upper right')

# ax.plot(x, train_loss_4, label='Polynomial Order=4')
# ax.legend(loc='upper right')

ax.plot(x, val_loss_4, label='Polynomial Order=4')
ax.legend(loc='upper right')

ax.set_yscale('log')
ax.grid(True)
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Validation Loss')

# Show the figure
plt.show()