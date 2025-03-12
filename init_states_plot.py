import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Data
x = [-1000, -500, 0, 500, 1000]
y = [-2000, -1000, 0, 1000, 2000]

# Load airplane image
ego_img = plt.imread("ego.png")  # Replace with the path to your airplane image
int_img = plt.imread('intruder.png')

# Plot
fig, ax = plt.subplots()
ax.scatter(x, y, s=0)  # Placeholder for coordinates

x_int = -2000
y_int = 0

imagebox = OffsetImage(int_img, zoom=0.1)  # Adjust `zoom` for size
ab = AnnotationBbox(imagebox, (x_int, y_int), frameon=False)
ax.add_artist(ab)

# Add airplane image at each point
for xi in x:
    for yi in y:
        imagebox = OffsetImage(ego_img, zoom=0.1)  # Adjust `zoom` for size
        ab = AnnotationBbox(imagebox, (xi, yi), frameon=False)
        ax.add_artist(ab)

plt.xlabel("x (ft)")
plt.ylabel("y (ft)")
plt.xlim((-2500, 1500))
plt.show()
