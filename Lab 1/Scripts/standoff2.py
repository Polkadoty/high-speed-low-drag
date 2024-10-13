import cv2
import numpy as np
import matplotlib.pyplot as plt

def onclick(event):
    global points
    if event.button == 1 and len(points) < 3:  # Left mouse button
        points.append((event.xdata, event.ydata))
        plt.plot(event.xdata, event.ydata, 'ro')
        plt.draw()
        if len(points) == 3:
            plt.close()

def find_sphere_center_and_radius(top, bottom):
    center_x = (top[0] + bottom[0]) / 2
    center_y = (top[1] + bottom[1]) / 2
    radius = np.sqrt((top[0] - bottom[0])**2 + (top[1] - bottom[1])**2) / 2
    return (center_x, center_y), radius

def closest_point_on_circle(center, radius, point):
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    distance = np.sqrt(dx**2 + dy**2)
    return (
        center[0] + radius * dx / distance,
        center[1] + radius * dy / distance
    )

def measure_shock_standoff(image_path, calibration_factor):
    global points
    points = []

    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image and wait for point selection
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_rgb)
    ax.set_title("Click to select: 1) Sphere top, 2) Sphere bottom, 3) Shock wave point")
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(points) != 3:
        print("Error: Three points were not selected.")
        return None

    # Find sphere center and radius
    top, bottom, shock = points
    center, radius = find_sphere_center_and_radius(top, bottom)

    # Find closest point on circle to shock point
    closest_point = closest_point_on_circle(center, radius, shock)

    # Calculate distances
    shock_distance = np.sqrt((shock[0] - center[0])**2 + (shock[1] - center[1])**2)
    standoff_distance_px = shock_distance - radius
    
    # Calculate measurements
    standoff_distance_mm = standoff_distance_px / calibration_factor
    sphere_diameter_mm = (2 * radius) / calibration_factor

    # Visualize the measurement
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_rgb)
    circle = plt.Circle(center, radius, color='r', fill=False)
    ax.add_artist(circle)
    ax.plot([closest_point[0], shock[0]], [closest_point[1], shock[1]], 'b-')
    ax.plot([p[0] for p in points], [p[1] for p in points], 'ro')
    ax.plot(center[0], center[1], 'go', markersize=10)
    ax.set_title(f"Shock Standoff Distance: {standoff_distance_mm:.2f} mm\n"
                 f"Sphere Diameter: {sphere_diameter_mm:.2f} mm")
    plt.show()

    return standoff_distance_mm, sphere_diameter_mm

# Usage
image_path = "Lab 1\Data\Images\M1_75_horizontal.bmp"
calibration_factor = 14.3  # pixels/mm, as you provided
standoff, diameter = measure_shock_standoff(image_path, calibration_factor)
if standoff is not None:
    print(f"Shock standoff distance: {standoff:.2f} mm")
    print(f"Sphere diameter: {diameter:.2f} mm")
