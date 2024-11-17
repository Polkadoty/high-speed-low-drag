import cv2
import numpy as np

def undistort_image(image_path, cal_factor_vertical, cal_factor_horizontal):
    # Read the image
    img = cv2.imread(image_path)
    
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Define the camera matrix
    camera_matrix = np.array([[cal_factor_horizontal, 0, w / 2],
                               [0, cal_factor_vertical, h / 2],
                               [0, 0, 1]], dtype=np.float32)
    
    # Assuming no distortion coefficients
    distortion_coeffs = np.zeros((4, 1))  # [k1, k2, p1, p2]

    # Undistort the image
    undistorted_img = cv2.undistort(img, camera_matrix, distortion_coeffs)

    # Display the undistorted image
    cv2.imshow("Undistorted Image", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    calibration_factors = (14.23, 14.45)  # Replace with actual calibration factors
    image_to_undistort = r"Lab 2\Images\cone_45deg_225.bmp"  # Path to the image to be undistorted
    cal_v, cal_h = calibration_factors
    
    undistort_image(image_to_undistort, cal_v, cal_h)
