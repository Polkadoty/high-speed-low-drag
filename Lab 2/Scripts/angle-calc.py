import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

class AngleMeasurer:
    def __init__(self):
        self.points = []
        self.image = None
        
    def calculate_angle(self, p1, p2, p3):
        """
        Calculate angle between three points
        p2 is the vertex point
        Returns angle in degrees
        """
        # Convert points to numpy arrays for vector calculations
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        # Create vectors from vertex to other points
        ba = a - b
        bc = c - b
        
        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def onclick(self, event):
        """Handle mouse clicks to collect points"""
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            self.points.append((event.xdata, event.ydata))
            
            # Plot point
            plt.plot(event.xdata, event.ydata, 'ro')
            
            # If this is second or third point, draw line
            if len(self.points) >= 2:
                plt.plot([self.points[-2][0], self.points[-1][0]], 
                        [self.points[-2][1], self.points[-1][1]], 'r-')
            
            plt.draw()
            
            # After collecting 3 points, calculate and display angle
            if len(self.points) == 3:
                angle = self.calculate_angle(self.points[0], self.points[1], self.points[2])
                plt.title(f'Measured Angle: {angle:.2f}°')
                plt.draw()
    
    def measure_angle(self, image_path):
        """Main function to load image and get angle measurement"""
        # Read image
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Convert BGR to RGB for matplotlib
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Create figure and connect click event
        plt.figure(figsize=(12, 8))
        plt.imshow(self.image)
        plt.title('Click 3 points to measure angle (middle point is vertex)')
        plt.connect('button_press_event', self.onclick)
        
        # Add instructions text box
        plt.figtext(0.02, 0.02, 
                   'Click 3 points:\n1. First point\n2. Vertex point\n3. Last point\nClose window when done',
                   bbox=dict(facecolor='white', alpha=0.7))
        
        plt.show()
        
        # Return measured angle if 3 points were selected
        if len(self.points) == 3:
            return self.calculate_angle(self.points[0], self.points[1], self.points[2])
        else:
            return None

def main():
    # Get folder path from user
    folder_path = input("Enter path to folder containing images: ")
    folder = Path(folder_path)
    
    if not folder.is_dir():
        print(f"Error: {folder_path} is not a valid directory.")
        return
    
    # Create list to store all measurements
    all_measurements = []
    
    # Loop through all image files in the folder
    for image_path in folder.glob("*.bmp"):  # Adjust the extension as needed
        try:
            # Create measurer and get angle
            measurer = AngleMeasurer()
            angle = measurer.measure_angle(image_path)
            
            if angle is not None:
                print(f"\nMeasured angle for {image_path.name}: {angle:.2f}°")
                
                # Store measurements in format needed for mach-analysis.py
                measurement = {
                    'run': image_path.stem,  # filename without extension
                    'shock_angle': angle,     # measured shock angle
                    'deflection_angle': None  # will be filled in with next measurement
                }
                all_measurements.append(measurement)
                
                # Optionally save individual measurements
                output_file = image_path.with_suffix('.txt')
                with open(output_file, 'w') as f:
                    f.write(f"Points:\n")
                    for i, point in enumerate(measurer.points):
                        f.write(f"Point {i+1}: ({point[0]:.2f}, {point[1]:.2f})\n")
                    f.write(f"\nMeasured angle: {angle:.2f}°")
                print(f"Results saved to: {output_file}")
                
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
    
    # Save all measurements to CSV
    if all_measurements:
        df = pd.DataFrame(all_measurements)
        csv_output = folder / 'angle_measurements.csv'
        df.to_csv(csv_output, index=False)
        print(f"\nAll measurements saved to: {csv_output}")
        print("\nNote: Please fill in the deflection_angle values in the CSV file.")

if __name__ == "__main__":
    main()