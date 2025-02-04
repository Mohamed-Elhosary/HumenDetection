import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# Load the background landscape image
image_path = "D:/Year4/Capstone/landscape.jpg"  # Replace with your actual image
bg_image = cv2.imread(image_path)
bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB

# Define grid size
rows, cols = 10, 10  # Define the grid resolution

# Create an empty heatmap (all zeros)
grid = np.zeros((rows, cols))

# Create a figure
fig, ax = plt.subplots(figsize=(bg_image.shape[1] / 100, bg_image.shape[0] / 100), dpi=100)

# Display the background image
ax.imshow(bg_image, extent=[0, cols, 0, rows])  # Scale the image to match the grid

# Display the heatmap overlay with a continuous colormap
heatmap = ax.imshow(grid, cmap="jet", alpha=0.5, extent=[0, cols, 0, rows], origin="upper", vmin=0, vmax=1)

# Add grid lines
ax.set_xticks(np.arange(0, cols, 1))
ax.set_yticks(np.arange(0, rows, 1))
ax.grid(color="black", linestyle="-", linewidth=0.5)

# Remove tick labels
ax.set_xticklabels([])
ax.set_yticklabels([])

# Add colorbar
cbar = plt.colorbar(heatmap)
cbar.set_label("Heatmap Intensity")

plt.ion()  # Turn on interactive mode

# Function to update the heatmap dynamically
def update_heatmap(row, col, value):
    global grid, heatmap

    # Update grid data with continuous values between 0 and 1
    grid[row, col] = value  

    # Clear the previous heatmap overlay
    ax.clear()
    
    # Redisplay the background image
    ax.imshow(bg_image, extent=[0, cols, 0, rows])

    # Redisplay the updated heatmap with the fixed colormap range
    heatmap = ax.imshow(grid, cmap="jet", alpha=0.3, extent=[0, cols, 0, rows], origin="upper", vmin=0, vmax=1)

    # Re-add grid lines
    ax.set_xticks(np.arange(0, cols, 1))
    ax.set_yticks(np.arange(0, rows, 1))
    ax.grid(color="black", linestyle="-", linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Redraw the figure
    plt.draw()
    plt.pause(0.1)  # Pause for visualization

# Function to traverse the grid in an S-path
def s_path_traversal():
    for i in range(rows):
        if i % 2 == 0:  # Even rows: left to right
            col_range = range(cols)
        else:  # Odd rows: right to left
            col_range = reversed(range(cols))
        
        for j in col_range:
            value = np.random.random()  # Generate a random continuous value between 0 and 1
            update_heatmap(i, j, value)
            time.sleep(0.1)  # Simulate processing delay

# Start the live update process
s_path_traversal()

plt.ioff()  # Turn off interactive mode
plt.show()  # Display the final state
