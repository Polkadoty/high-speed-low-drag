import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import cv2
import pandas as pd

class DatProcessor:
    def __init__(self):
        self.rect_coords = None
        self.image = None
        
    def load_dat_file(self, filename):
        """Load and reshape the .dat file into a 2D array"""
        # Read the data file, skipping the first 5 header lines
        data = np.loadtxt(filename, skiprows=4)
        
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
        
        rs = RectangleSelector(ax, self.onselect, interactive=True)
        
        plt.title('Draw a rectangle around the region of interest\nClose window when done')
        plt.show()
        
        return self.rect_coords
    
    def calculate_stats(self, rect):
        """Calculate mean and std dev for selected region"""
        x, y, w, h = [int(v) for v in rect]
        roi = self.image[y:y+h, x:x+w]
        
        Irat = np.mean(roi)
        dIrat = np.mean(np.std(roi, axis=0))
        
        return Irat, dIrat

def load_pressure_data(filename):
    """
    Load pressure and temperature data from Mach*.txt files
    """
    # Define column names
    columns = ['X_Value', 'Temperature', 'Stagnation', 'Static', 'Camera']
    
    # Load data
    data = pd.read_csv(filename, sep='\t', names=columns)
    return data

def calculate_mach(P0, Pi):
    """Calculate Mach number from pressure ratio"""
    return np.sqrt(5 * ((P0/Pi)**(2/7) - 1))

def process_pressure_data(data, start_idx=None, end_idx=None):
    """
    Process pressure data to get pressure values and Mach number
    """
    # Convert voltage to pressure (kPa)
    if start_idx is None or end_idx is None:
        # Use all data if no indices specified
        P0_voltage = data['Stagnation'].mean()
        Pi_voltage = data['Static'].mean()
    else:
        P0_voltage = data['Stagnation'][start_idx:end_idx].mean()
        Pi_voltage = data['Static'][start_idx:end_idx].mean()
    
    # Convert to pressure (kPa)
    P0 = 4137 * P0_voltage + 101.325
    Pi = 1034 * Pi_voltage + 101.325
    
    # Calculate Mach number
    M = calculate_mach(P0, Pi)
    
    return P0, Pi, M

def process_all_data(mach_files, tsp_files):
    """
    Process all pressure and TSP data files
    
    Parameters:
    -----------
    mach_files : dict
        Dictionary of Mach number to pressure data filename
    tsp_files : dict
        Dictionary of Mach number to list of TSP .dat filenames
    """
    results = {}
    
    for mach_num, pressure_file in mach_files.items():
        # Process pressure data
        pressure_data = load_pressure_data(pressure_file)
        P0, Pi, M = process_pressure_data(pressure_data)
        
        # Process TSP data for this Mach number
        tsp_results = []
        for tsp_file in tsp_files[mach_num]:
            processor = DatProcessor()
            try:
                processor.load_dat_file(tsp_file)
                rect = processor.get_rect_interactive()
                Irat, dIrat = processor.calculate_stats(rect)
                tsp_results.append({
                    'filename': tsp_file,
                    'Irat': Irat,
                    'dIrat': dIrat
                })
            except Exception as e:
                print(f"Error processing {tsp_file}: {e}")
        
        results[mach_num] = {
            'P0': P0,
            'Pi': Pi,
            'Mach': M,
            'TSP_data': tsp_results
        }
    
    return results

if __name__ == "__main__":
    # Example usage
    mach_files = {
        2.0: "Mach2.txt",
        2.5: "Mach25.txt",
        3.0: "Mach3.txt"
    }
    
    tsp_files = {
        2.0: [f"M2_{i}.dat" for i in range(1, 4)],
        2.5: [f"M2.5_{i}.dat" for i in range(1, 4)],
        3.0: [f"M3_{i}.dat" for i in range(1, 4)]
    }
    
    try:
        results = process_all_data(mach_files, tsp_files)
        
        # Print results
        for mach_num, data in results.items():
            print(f"\nMach {mach_num}:")
            print(f"P0: {data['P0']:.2f} kPa")
            print(f"Pi: {data['Pi']:.2f} kPa")
            print(f"Calculated Mach: {data['Mach']:.2f}")
            print("\nTSP Results:")
            for tsp_result in data['TSP_data']:
                print(f"  {tsp_result['filename']}:")
                print(f"    Irat: {tsp_result['Irat']:.6f}")
                print(f"    dIrat: {tsp_result['dIrat']:.6f}")
    
    except Exception as e:
        print(f"Error processing data: {e}")