import cv2
import numpy as np
import matplotlib.pyplot as plt

def onclick(event):
    global points_vertical, points_horizontal, current_phase
    if event.button == 1:  # Left mouse button
        if current_phase == 'vertical' and len(points_vertical) < 2:
            points_vertical.append((event.xdata, event.ydata))
            plt.plot(event.xdata, event.ydata, 'ro')
            plt.draw()
            if len(points_vertical) == 2:
                plt.close()
        elif current_phase == 'horizontal' and len(points_horizontal) < 2:
            points_horizontal.append((event.xdata, event.ydata))
            plt.plot(event.xdata, event.ydata, 'bo')
            plt.draw()
            if len(points_horizontal) == 2:
                plt.close()

def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculate_calibration_factor(image_path, grid_spacing_mm=5):
    global points_vertical, points_horizontal, current_phase
    points_vertical = []
    points_horizontal = []
    
    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Vertical calibration
    current_phase = 'vertical'
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_title("Click on two points 10 grid spaces apart VERTICALLY (red)")
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Horizontal calibration
    current_phase = 'horizontal'
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_title("Click on two points 10 grid spaces apart HORIZONTALLY (blue)")
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(points_vertical) != 2 or len(points_horizontal) != 2:
        print("Error: Points were not selected properly.")
        return None

    # Calculate distances
    distance_px_vertical = calculate_distance(points_vertical[0], points_vertical[1])
    distance_px_horizontal = calculate_distance(points_horizontal[0], points_horizontal[1])

    # Calculate calibration factors (pixels per mm)
    cal_factor_vertical = distance_px_vertical / (10 * grid_spacing_mm)
    cal_factor_horizontal = distance_px_horizontal / (10 * grid_spacing_mm)

    # Calculate distortion ratio (should be 1.0 if no distortion)
    distortion_ratio = cal_factor_vertical / cal_factor_horizontal

    # Visualize the selected points and measurements
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    
    # Plot vertical measurement
    ax.plot([points_vertical[0][0], points_vertical[1][0]], 
            [points_vertical[0][1], points_vertical[1][1]], 'r-')
    ax.plot([p[0] for p in points_vertical], [p[1] for p in points_vertical], 'ro')
    
    # Plot horizontal measurement
    ax.plot([points_horizontal[0][0], points_horizontal[1][0]], 
            [points_horizontal[0][1], points_horizontal[1][1]], 'b-')
    ax.plot([p[0] for p in points_horizontal], [p[1] for p in points_horizontal], 'bo')
    
    ax.set_title(f"Calibration Factors:\nVertical: {cal_factor_vertical:.2f} px/mm\n" +
                 f"Horizontal: {cal_factor_horizontal:.2f} px/mm\n" +
                 f"Distortion ratio (V/H): {distortion_ratio:.3f}")
    plt.show()

    return cal_factor_vertical, cal_factor_horizontal, distortion_ratio

# Usage
calibration_image_path = r"Lab 2\Images\cone_calibration_img.bmp"
calibration_results = calculate_calibration_factor(calibration_image_path)
if calibration_results:
    cal_v, cal_h, dist_ratio = calibration_results
    print(f"Vertical calibration factor: {cal_v:.2f} pixels/mm")
    print(f"Horizontal calibration factor: {cal_h:.2f} pixels/mm")
    print(f"Distortion ratio (V/H): {dist_ratio:.3f}")
