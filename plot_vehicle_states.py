import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Create data for two vehicles
#t = np.linspace(0, 10, 100)  # Time steps
vehicle1_x = np.load()  # Vehicle 1 x-coordinate
vehicle1_y = np.cos(t)  # Vehicle 1 y-coordinate
vehicle2_x = np.sin(t + np.pi / 2)  # Vehicle 2 x-coordinate
vehicle2_y = np.cos(t + np.pi / 2)  # Vehicle 2 y-coordinate

# Initialize the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Vehicle Movement")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")

# Plot the paths of the vehicles
line1, = ax.plot([], [], 'b-', label="Vehicle 1 Path", lw=1)
line2, = ax.plot([], [], 'r-', label="Vehicle 2 Path", lw=1)

# Markers for vehicles
vehicle1_marker, = ax.plot([], [], 'bo', label="Vehicle 1", markersize=8)
vehicle2_marker, = ax.plot([], [], 'ro', label="Vehicle 2", markersize=8)

# Add a legend
ax.legend()

# Initialization function
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    vehicle1_marker.set_data([], [])
    vehicle2_marker.set_data([], [])
    return line1, line2, vehicle1_marker, vehicle2_marker

# Animation function
def update(frame):
    # Update the paths
    line1.set_data(vehicle1_x[:frame], vehicle1_y[:frame])
    line2.set_data(vehicle2_x[:frame], vehicle2_y[:frame])
    
    # Update the markers
    vehicle1_marker.set_data(vehicle1_x[frame], vehicle1_y[frame])
    vehicle2_marker.set_data(vehicle2_x[frame], vehicle2_y[frame])
    
    return line1, line2, vehicle1_marker, vehicle2_marker

# Create the animation
frames = len(t)
anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

# Save as a GIF
anim.save("vehicle_movement.gif", writer=PillowWriter(fps=10))

# Show the animation (optional)
plt.show()
