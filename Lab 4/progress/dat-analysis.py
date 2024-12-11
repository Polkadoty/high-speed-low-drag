import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import cv2

class DatProcessor:
    def __init__(self):
        self.rect_coords = None
        self.image = None
        
    def load_dat_file(self, filename):
        """Load and reshape the .dat file into a 2D array"""
        # Read the data file, skipping the first 5 header lines
        data = np.loadtxt(filename, skiprows=4)  # Adjusted to skip 5 header lines
        
        # Extract the third column (index 2)
        values = data[:, 2]
        
        # Check if the number of values is sufficient for reshaping
        if len(values) < 600 * 800:
            raise ValueError(f"Not enough data to reshape into 600x800 array. Found {len(values)} values.")
        
        # Reshape into 600x800 array
        B = np.zeros((600, 800))
        counter = 0
        
        for x in range(800):
            for y in range(600):
                B[y, x] = values[counter]
                counter += 1
                
        # Rotate image -90 degrees
        self.image = cv2.rotate(B, cv2.ROTATE_90_CLOCKWISE)
        return self.image
    
    def onselect(self, eclick, erelease):
        """Callback for rectangle selection"""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.rect_coords = [min(x1, x2), min(y1, y2), 
                          abs(x2 - x1), abs(y2 - y1)]
        
    def get_rect_interactive(self):
        """Display image and get rectangle selection interactively"""
        fig, ax = plt.subplots()
        ax.imshow(self.image)
        
        # Create rectangle selector
        rs = RectangleSelector(ax, self.onselect, interactive=True)
        
        plt.title('Draw a rectangle around the region of interest\nClose window when done')
        plt.show()
        
        return self.rect_coords
    
    def calculate_stats(self, rect):
        """Calculate mean and std dev for selected region"""
        x, y, w, h = [int(v) for v in rect]
        roi = self.image[y:y+h, x:x+w]
        
        Irat = np.mean(roi)
        dIrat = np.mean(np.std(roi, axis=0))  # Mean of std devs along columns
        
        return Irat, dIrat

def process_dat_file(filename):
    """Main function to process a .dat file"""
    processor = DatProcessor()
    
    # Load and process the file
    processor.load_dat_file(filename)
    
    # Get rectangle selection
    rect = processor.get_rect_interactive()
    
    # Calculate statistics
    Irat, dIrat = processor.calculate_stats(rect)
    
    print(f"Irat: {Irat:.6f}")
    print(f"dIrat: {dIrat:.6f}")
    
    return Irat, dIrat

if __name__ == "__main__":
    # Example usage
    filename = r"Lab 4/F24 Lab 4 Data/M2_1.dat"  # Using forward slashes
    try:
        Irat, dIrat = process_dat_file(filename)
    except ValueError as e:
        print(f"Error processing file {filename}: {e}")
    
    # To process multiple files:
    # files = ["M2_1.dat", "M2.5_1.dat", "M3_1.dat"]
    # results = {f: process_dat_file(f) for f in files}