import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import fsolve, curve_fit

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

def calculate_reynolds_numbers(mach_number, pressure, temperature, diameter):
    # Constants
    gamma = 1.4  # Ratio of specific heats for air
    R = 287.05  # Gas constant for air in J/(kg·K)
    
    # Sutherland's law constants
    C = 120  # Sutherland's constant for air in K
    T0 = 291.15  # Reference temperature in K
    mu0 = 1.827e-5  # Reference viscosity in Pa·s

    # Calculate temperature ratio
    T_ratio = 1 + (gamma - 1) / 2 * mach_number**2
    T = temperature * T_ratio  # Static temperature

    # Calculate density
    rho = pressure / (R * T)

    # Calculate velocity
    V = mach_number * np.sqrt(gamma * R * T)

    # Calculate viscosity using Sutherland's law
    mu = mu0 * (T / T0)**(3/2) * (T0 + C) / (T + C)

    # Calculate Reynolds numbers
    Re_unit = rho * V / mu
    Re_D = Re_unit * diameter

    return Re_unit, Re_D

def mach_number_uncertainty(p0, p, dp0, dp, gamma=1.4):
    # Function to solve for Mach number
    def mach_equation(M):
        return (p0/p) - (1 + (gamma-1)/2 * M**2)**(gamma/(gamma-1))

    # Calculate Mach number
    M = fsolve(mach_equation, 1.0)[0]

    # Partial derivatives
    dM_dp0 = M / (2*p0) * (1 + (gamma-1)/2 * M**2)
    dM_dp = -M / (2*p) * (1 + (gamma-1)/2 * M**2)

    # Uncertainty propagation
    dM = np.sqrt((dM_dp0 * dp0)**2 + (dM_dp * dp)**2)

    return M, dM

def process_file(file_path):
    data = read_data_file(file_path)
    
    stagnation_pressure = voltage_to_pressure(data[:, 0], 60)
    static_pressure = voltage_to_pressure(data[:, 1], 15)
    
    start, end = find_steady_state(stagnation_pressure)
    
    avg_stagnation_pressure = np.mean(stagnation_pressure[start:end])
    avg_static_pressure = np.mean(static_pressure[start:end])
    
    mach_number, mach_uncertainty = mach_number_uncertainty(avg_stagnation_pressure + 14.7, 
                                                            avg_static_pressure + 14.7, 
                                                            0.01 * avg_stagnation_pressure, 
                                                            0.01 * avg_static_pressure)
    
    # Calculate Reynolds numbers
    diameter = 0.0163  # Sphere diameter in meters
    Re_unit, Re_D = calculate_reynolds_numbers(mach_number, avg_static_pressure * 6894.75729,  # Convert psi to Pa
                                               297, diameter)  # Assuming 297 K (24°C) ambient temperature
    
    # Calculate Reynolds number uncertainty (simplified, assuming only Mach number contributes significantly)
    dRe_D = Re_D * mach_uncertainty / mach_number
    
    # Visualize the data
    # plt.figure(figsize=(12, 6))
    # plt.plot(stagnation_pressure, label='Stagnation Pressure')
    # plt.plot(static_pressure, label='Static Pressure')
    # plt.axvline(start, color='r', linestyle='--', label='Steady State Start')
    # plt.axvline(end, color='r', linestyle='--', label='Steady State End')
    # plt.axhline(avg_stagnation_pressure, color='g', linestyle=':', label='Avg Stagnation')
    # plt.axhline(avg_static_pressure, color='m', linestyle=':', label='Avg Static')
    # plt.legend()
    # plt.title(f'Pressure Data for {os.path.basename(file_path)}')
    # plt.xlabel('Sample')
    # plt.ylabel('Pressure (psi)')
    # plt.show()
    
    print(f"File: {os.path.basename(file_path)}")
    print(f"Stagnation Pressure (gauge): {avg_stagnation_pressure:.2f} psi")
    print(f"Static Pressure (gauge): {avg_static_pressure:.2f} psi")
    print(f"Stagnation Pressure (absolute): {avg_stagnation_pressure + 14.7:.2f} psi")
    print(f"Static Pressure (absolute): {avg_static_pressure + 14.7:.2f} psi")
    print(f"Pressure Ratio (p/p0): {(avg_static_pressure + 14.7) / (avg_stagnation_pressure + 14.7):.4f}")
    print(f"Calculated Mach Number: {mach_number:.2f}")
    print(f"Unit Reynolds Number: {Re_unit:.2e} 1/m")
    print(f"Diametric Reynolds Number: {Re_D:.2e}")
    print(f"Reynolds Number Uncertainty: {dRe_D:.2e}")
    print("---")
    
    return mach_number, mach_uncertainty, avg_stagnation_pressure, avg_static_pressure, Re_unit, Re_D, dRe_D

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
            calculated_mach, mach_uncertainty, stagnation_p, static_p, Re_unit, Re_D, dRe_D = process_file(file_path)
            results.append((expected_mach, calculated_mach, mach_uncertainty, stagnation_p, static_p, Re_unit, Re_D, dRe_D))
            print(f"Processed {filename}: Expected M={expected_mach:.2f}, Calculated M={calculated_mach:.2f} ± {mach_uncertainty:.2f}")

# Sort results by expected Mach number
results.sort(key=lambda x: x[0])

# Plotting Mach number vs Re_D with error bars (vertical only)
plt.figure(figsize=(10, 6))
mach_numbers, re_d_numbers, re_d_uncertainties = zip(*[(calc_mach, re_d, dre_d) 
                                                       for _, calc_mach, _, _, _, _, re_d, dre_d in results])

# Use absolute values for uncertainties
re_d_uncertainties = np.abs(re_d_uncertainties)

plt.errorbar(mach_numbers, np.abs(re_d_numbers), yerr=re_d_uncertainties, fmt='o', capsize=5)
plt.xlabel('Mach Number')
plt.ylabel('Reynolds Number (Re_D)')
plt.title('Mach Number vs Reynolds Number with Uncertainties')
plt.grid(True)
plt.show()

# Print results
print("\nMach Number and Reynolds Number Comparison with Uncertainties:")
print("Expected M | Calculated M ± Uncertainty | Re_D ± Uncertainty")
print("-" * 70)
for expected, calculated, mach_unc, _, _, _, re_d, dre_d in results:
    print(f"{expected:.2f}       | {calculated:.2f} ± {abs(mach_unc):.2f}              | {re_d:.2e} ± {abs(dre_d):.2e}")

# Calculate and print average error
valid_results = [(exp, calc) for exp, calc, _, _, _, _, _, _ in results if not np.isnan(calc)]
if valid_results:
    errors = [abs(calc - exp) for exp, calc in valid_results]
    avg_error = np.mean(errors)
    print(f"\nAverage Mach number error: {avg_error:.4f}")
else:
    print("\nNo valid Mach number calculations.")

# Add these new functions after the existing functions

def calculate_nondimensional_standoff(mach_numbers, diameters, standoff_distances):
    return np.array(standoff_distances) / np.array(diameters)

def qualitative_scaling(M, c, gamma=1.4):
    return c * np.sqrt((1 + (gamma - 1) / 2 * M**2) / (M - 1))

def basic_fit(M, c, alpha, beta, gamma=1.4):
    return c * gamma**alpha * M**beta

def offset_fit(M, c, alpha, beta, gamma=1.4):
    return c * gamma**alpha * (M - 1)**beta

def plot_nondimensional_standoff(mach_numbers, nondimensional_standoff):
    plt.figure(figsize=(10, 6))
    plt.scatter(mach_numbers, nondimensional_standoff)
    plt.xlabel('Mach Number')
    plt.ylabel('δ/D')
    plt.title('Non-dimensional Standoff Distance vs Mach Number')
    plt.grid(True)
    plt.show()

def perform_curve_fits(mach_numbers, nondimensional_standoff):
    # Qualitative scaling fit
    popt_qual, _ = curve_fit(qualitative_scaling, mach_numbers, nondimensional_standoff)
    
    # Basic fit
    popt_basic, _ = curve_fit(basic_fit, mach_numbers, nondimensional_standoff)
    
    # Offset fit
    popt_offset, _ = curve_fit(offset_fit, mach_numbers, nondimensional_standoff)
    
    return popt_qual, popt_basic, popt_offset

def plot_curve_fits(mach_numbers, nondimensional_standoff, popt_qual, popt_basic, popt_offset):
    plt.figure(figsize=(12, 8))
    plt.scatter(mach_numbers, nondimensional_standoff, label='Data')
    
    M_fit = np.linspace(min(mach_numbers), max(mach_numbers), 100)
    
    plt.plot(M_fit, qualitative_scaling(M_fit, *popt_qual), 'r-', label='Qualitative Scaling')
    plt.plot(M_fit, basic_fit(M_fit, *popt_basic), 'g-', label='Basic Fit')
    plt.plot(M_fit, offset_fit(M_fit, *popt_offset), 'b-', label='Offset Fit')
    
    plt.xlabel('Mach Number')
    plt.ylabel('δ/D')
    plt.title('Curve Fits for Non-dimensional Standoff Distance')
    plt.legend()
    plt.grid(True)
    plt.show()

# ... (keep all existing code up to the standoff_data dictionary)

# Replace the first instance of standoff_data with this:
standoff_data = {
    2.25: [16.65, 16.82, 16.65, 2.40, 2.29, 2.44],
    1.75: [16.69, 16.65, 16.69, 3.95, 3.97, 4.11],
    2.0: [16.87, 16.51, 16.74, 2.80, 2.95, 2.98],
    2.5: [16.83, 16.69, 16.82, 2.02, 2.22, 2.07],
    2.75: [16.78, 16.78, 16.78, 2.00, 1.95, 1.93],
    3.0: [16.69, 16.78, 16.78, 1.71, 1.80, 1.74]
}

# Add these new functions after the existing functions

def calculate_nondimensional_standoff(mach_numbers, diameters, standoff_distances):
    return np.array(standoff_distances) / np.array(diameters)

def qualitative_scaling(M, c, gamma=1.4):
    return c * np.sqrt((1 + (gamma - 1) / 2 * M**2) / (M - 1))

def basic_fit(M, c, alpha, beta, gamma=1.4):
    return c * gamma**alpha * M**beta

def offset_fit(M, c, alpha, beta, gamma=1.4):
    return c * gamma**alpha * (M - 1)**beta

def plot_nondimensional_standoff(mach_numbers, nondimensional_standoff):
    plt.figure(figsize=(10, 6))
    plt.scatter(mach_numbers, nondimensional_standoff)
    plt.xlabel('Mach Number')
    plt.ylabel('δ/D')
    plt.title('Non-dimensional Standoff Distance vs Mach Number')
    plt.grid(True)
    plt.show()

def perform_curve_fits(mach_numbers, nondimensional_standoff):
    # Qualitative scaling fit
    popt_qual, _ = curve_fit(qualitative_scaling, mach_numbers, nondimensional_standoff)
    
    # Basic fit
    popt_basic, _ = curve_fit(basic_fit, mach_numbers, nondimensional_standoff)
    
    # Offset fit
    popt_offset, _ = curve_fit(offset_fit, mach_numbers, nondimensional_standoff)
    
    return popt_qual, popt_basic, popt_offset

def plot_curve_fits(mach_numbers, nondimensional_standoff, popt_qual, popt_basic, popt_offset):
    plt.figure(figsize=(12, 8))
    plt.scatter(mach_numbers, nondimensional_standoff, label='Data')
    
    M_fit = np.linspace(min(mach_numbers), max(mach_numbers), 100)
    
    plt.plot(M_fit, qualitative_scaling(M_fit, *popt_qual), 'r-', label='Qualitative Scaling')
    plt.plot(M_fit, basic_fit(M_fit, *popt_basic), 'g-', label='Basic Fit')
    plt.plot(M_fit, offset_fit(M_fit, *popt_offset), 'b-', label='Offset Fit')
    
    plt.xlabel('Mach Number')
    plt.ylabel('δ/D')
    plt.title('Curve Fits for Non-dimensional Standoff Distance')
    plt.legend()
    plt.grid(True)
    plt.show()

# ... (keep the rest of your existing code)

# After your existing code, add:

# Process the standoff data
mach_numbers = list(standoff_data.keys())
diameters = [[d for d in data[:3]] for data in standoff_data.values()]
standoff_distances = [[s for s in data[3:]] for data in standoff_data.values()]

# Calculate average values
avg_diameters = [np.mean(d) for d in diameters]
avg_standoffs = [np.mean(s) for s in standoff_distances]

# Calculate non-dimensional standoff
nondimensional_standoff = calculate_nondimensional_standoff(mach_numbers, avg_diameters, avg_standoffs)

# Plot non-dimensional standoff distance
plot_nondimensional_standoff(mach_numbers, nondimensional_standoff)

# Perform curve fits
popt_qual, popt_basic, popt_offset = perform_curve_fits(mach_numbers, nondimensional_standoff)

# Plot curve fits
plot_curve_fits(mach_numbers, nondimensional_standoff, popt_qual, popt_basic, popt_offset)

# Print table of fit constants
print("\nFit Constants:")
print("Qualitative Scaling: c =", popt_qual[0])
print("Basic Fit: c =", popt_basic[0], ", α =", popt_basic[1], ", β =", popt_basic[2])
print("Offset Fit: c =", popt_offset[0], ", α =", popt_offset[1], ", β =", popt_offset[2])