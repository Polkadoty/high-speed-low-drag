import cv2
import numpy as np
import matplotlib.pyplot as plt

def onclick(event):
    global points
    if event.button == 1 and len(points) < 2:  # Left mouse button
        points.append((event.xdata, event.ydata))
        plt.plot(event.xdata, event.ydata, 'ro')
        plt.draw()
        if len(points) == 2:
            plt.close()

def calculate_calibration_factor(image_path, grid_spacing_mm=5):
    global points
    points = []

    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image and wait for point selection
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_title("Click on two points 10 grid spaces apart")
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(points) != 2:
        print("Error: Two points were not selected.")
        return None

    # Calculate distance between selected points
    distance_px = np.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)

    # Calculate calibration factor (pixels per mm)
    calibration_factor = distance_px / (10 * grid_spacing_mm)

    # Visualize the selected points
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], 'r-')
    ax.plot([p[0] for p in points], [p[1] for p in points], 'ro')
    ax.set_title(f"Calibration: {calibration_factor:.2f} pixels/mm")
    plt.show()

    return calibration_factor

# Usage
calibration_image_path = "Lab 1\Data\Images\Wind_off_calibrate.bmp"
cal_factor = calculate_calibration_factor(calibration_image_path)
if cal_factor:
    print(f"Calibration factor: {cal_factor:.2f} pixels/mm")