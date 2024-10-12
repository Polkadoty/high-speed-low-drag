import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def read_data_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data_start = next(i for i, line in enumerate(lines) if "X_Value" in line)
    data = np.genfromtxt(lines[data_start+1:], delimiter='\t', usecols=(1, 2))
    return data

def voltage_to_pressure(voltage, conversion_factor):
    return voltage * (conversion_factor / 0.1)

def find_steady_state(data, window_size=50):
    # Find the index of the maximum pressure
    peak_index = np.argmax(np.abs(data))
    
    # Define a region around the peak to search for the steady state
    search_start = max(0, peak_index - window_size*2)
    search_end = min(len(data), peak_index + window_size*2)
    
    # Calculate moving standard deviation
    std_dev = np.array([np.std(data[i:i+window_size]) for i in range(search_start, search_end-window_size)])
    
    # Find the region with the lowest standard deviation (most stable)
    stable_start = search_start + np.argmin(std_dev)
    
    return stable_start, stable_start + window_size

def calculate_mach_number(p0_gauge, p_gauge, p_atm=14.7, gamma=1.4):
    # Convert gauge pressures to absolute pressures
    p0 = p0_gauge + p_atm
    p = p_gauge + p_atm
    
    if p0 <= p or p <= 0:
        return np.nan
    
    # Use the correct isentropic flow equation for Mach number
    mach = np.sqrt((2 / (gamma - 1)) * ((p0 / p)**((gamma - 1) / gamma) - 1))
    
    return mach

def process_file(file_path):
    data = read_data_file(file_path)
    
    stagnation_pressure = voltage_to_pressure(data[:, 0], 60)
    static_pressure = voltage_to_pressure(data[:, 1], 15)
    
    start, end = find_steady_state(stagnation_pressure)
    
    avg_stagnation_pressure = np.mean(stagnation_pressure[start:end])
    avg_static_pressure = np.mean(static_pressure[start:end])
    
    mach_number = calculate_mach_number(avg_stagnation_pressure, avg_static_pressure)
    
    # Visualize the data
    plt.figure(figsize=(12, 6))
    plt.plot(stagnation_pressure, label='Stagnation Pressure')
    plt.plot(static_pressure, label='Static Pressure')
    plt.axvline(start, color='r', linestyle='--', label='Steady State Start')
    plt.axvline(end, color='r', linestyle='--', label='Steady State End')
    plt.axhline(avg_stagnation_pressure, color='g', linestyle=':', label='Avg Stagnation')
    plt.axhline(avg_static_pressure, color='m', linestyle=':', label='Avg Static')
    plt.legend()
    plt.title(f'Pressure Data for {os.path.basename(file_path)}')
    plt.xlabel('Sample')
    plt.ylabel('Pressure (psi)')
    plt.show()
    
    print(f"File: {os.path.basename(file_path)}")
    print(f"Stagnation Pressure (gauge): {avg_stagnation_pressure:.2f} psi")
    print(f"Static Pressure (gauge): {avg_static_pressure:.2f} psi")
    print(f"Stagnation Pressure (absolute): {avg_stagnation_pressure + 14.7:.2f} psi")
    print(f"Static Pressure (absolute): {avg_static_pressure + 14.7:.2f} psi")
    print(f"Pressure Ratio (p/p0): {(avg_static_pressure + 14.7) / (avg_stagnation_pressure + 14.7):.4f}")
    print(f"Calculated Mach Number: {mach_number:.2f}")
    print("---")
    
    return mach_number, avg_stagnation_pressure, avg_static_pressure



def extract_mach_number(filename):
    match = re.search(r'M(\d+)_(\d+)', filename)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return None

# Directory containing the data files
data_dir = r"Lab 1\Working-data"

results = []

for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_dir, filename)
        expected_mach = extract_mach_number(filename)
        if expected_mach:
            calculated_mach, stagnation_p, static_p = process_file(file_path)
            results.append((expected_mach, calculated_mach, stagnation_p, static_p))
            print(f"Processed {filename}: Expected M={expected_mach:.2f}, Calculated M={calculated_mach:.2f}")

# Sort results by expected Mach number
results.sort(key=lambda x: x[0])

# Plotting
plt.figure(figsize=(10, 6))
expected_mach, calculated_mach, _, _ = zip(*results)
plt.plot([1.5, 3.5], [1.5, 3.5], 'k--', label='Ideal')
plt.scatter(expected_mach, calculated_mach, color='blue', label='Calculated')
plt.xlabel('Expected Mach Number')
plt.ylabel('Calculated Mach Number')
plt.title('Expected vs Calculated Mach Number')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.xlim(1.5, 3.5)
plt.ylim(1.5, 3.5)
plt.show()

# Print results
print("\nMach Number Comparison:")
print("Expected | Calculated | Stagnation P (psi) | Static P (psi)")
print("-" * 60)
for expected, calculated, stagnation_p, static_p in results:
    print(f"{expected:.2f}     | {calculated:.2f}       | {stagnation_p:.2f}             | {static_p:.2f}")

# Calculate and print average error
valid_results = [(exp, calc) for exp, calc, _, _ in results if not np.isnan(calc)]
if valid_results:
    errors = [abs(calc - exp) for exp, calc in valid_results]
    avg_error = np.mean(errors)
    print(f"\nAverage Mach number error: {avg_error:.4f}")
else:
    print("\nNo valid Mach number calculations.")