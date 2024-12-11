import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
import pandas as pd
from matplotlib.widgets import LassoSelector
from scipy.optimize import fsolve

class ObliqueShockCalculator:
    def __init__(self):
        self.gamma = 1.4
        self.theta = np.deg2rad(15)
        
    def find_beta(self, M1):
        """Find shock angle beta using theta-beta-M relationship"""
        def theta_beta_M(beta):
            # Theta-beta-M equation
            return np.tan(self.theta) - 2 * (1/np.tan(beta)) * \
                   ((M1**2 * np.sin(beta)**2 - 1)/(M1**2 * (self.gamma + np.cos(2*beta)) + 2))
        
        # Initial guess for beta (slightly larger than theta)
        beta_guess = self.theta * 1.1
        
        # Solve for beta
        beta = fsolve(theta_beta_M, beta_guess)[0]
        
        return beta

    def get_steady_state_pressures(self, mach_num):
        """Get steady state pressure values from data files"""
        if isinstance(mach_num, float) and mach_num == 2.5:
            filename = f'Lab 4/F24 Lab 4 Data/Mach25.txt'
        else:
            filename = f'Lab 4/F24 Lab 4 Data/Mach{int(mach_num)}.txt'
            
        # Read data
        data = pd.read_csv(filename, sep='\t',
                          names=['Time', 'Temperature', 'Stagnation', 'Static', 'Camera'])
        
        # Convert pressures to kPa (these conversion factors already give kPa)
        P0 = 4137 * data['Stagnation'] + 101.325  # Already in kPa
        Pi = 1034 * data['Static'] + 101.325      # Already in kPa
        
        # Get peak stagnation pressure and corresponding static pressure
        peak_idx = np.argmax(P0)
        steady_P0 = P0[peak_idx]
        steady_Pi = Pi[peak_idx]
        
        return steady_P0, steady_Pi

    def calculate_shock_properties(self, M1, selected_temp, P1, dM1=0.05, dT=2.3, dP=6.1):
        """Calculate properties behind oblique shock with uncertainties using sequential perturbation
        
        Uncertainties:
        - dM1: Mach number uncertainty (±0.05 typical for supersonic wind tunnels)
        - dT: Temperature uncertainty (±5K from TSP measurements)
        - dP: Pressure uncertainty (±5kPa from transducer calibration)
        """
        # Base calculation
        base = self._calculate_properties(M1, selected_temp, P1)
        
        # M1 perturbation
        high_M = self._calculate_properties(M1 + dM1, selected_temp, P1)
        low_M = self._calculate_properties(M1 - dM1, selected_temp, P1)
        
        # Temperature perturbation
        high_T = self._calculate_properties(M1, selected_temp + dT, P1)
        low_T = self._calculate_properties(M1, selected_temp - dT, P1)
        
        # Pressure perturbation
        high_P = self._calculate_properties(M1, selected_temp, P1 + dP)
        low_P = self._calculate_properties(M1, selected_temp, P1 - dP)
        
        # Calculate uncertainties using sequential perturbation
        uncertainties = {}
        for key in base.keys():
            if key != 'beta_deg':  # Skip angle for now
                dM = (high_M[key] - low_M[key])/2
                dT = (high_T[key] - low_T[key])/2
                dP = (high_P[key] - low_P[key])/2
                uncertainties[key] = np.sqrt(dM**2 + dT**2 + dP**2)
        
        # Add uncertainties to results
        results = base.copy()
        results['uncertainties'] = uncertainties
        
        return results

    def _calculate_properties(self, M1, selected_temp, P1):
        """Helper method for base property calculations (moved from calculate_shock_properties)"""
        beta = self.find_beta(M1)
        
        # Normal component of M1
        M1n = M1 * np.sin(beta)
        
        # Calculate ratios across normal shock
        P2_P1 = (2*self.gamma*M1n**2 - (self.gamma-1))/(self.gamma+1)
        T2_T1 = ((2*self.gamma*M1n**2 - (self.gamma-1))*((self.gamma-1)*M1n**2 + 2))/((self.gamma+1)**2 * M1n**2)
        rho2_rho1 = (self.gamma+1)*M1n**2/((self.gamma-1)*M1n**2 + 2)
        
        # Calculate M2n
        M2n_squared = ((self.gamma-1)*M1n**2 + 2)/(2*self.gamma*M1n**2 - (self.gamma-1))
        M2n = np.sqrt(M2n_squared)
        
        # Calculate M2 (downstream Mach number)
        M2 = M2n/np.sin(beta - self.theta)
        
        # Calculate absolute values (P1 is already in kPa)
        P2 = P2_P1 * P1  # Result in kPa since P1 is in kPa
        T2 = selected_temp
        rho2 = (P2 * 1000)/(287*T2)  # Convert P2 to Pa for density calculation
        
        # Calculate velocity components
        a1 = np.sqrt(self.gamma * 287 * T2)  # Speed of sound
        V1 = M1 * a1
        V2 = M2 * a1
        
        # Calculate stagnation conditions (P2 is in kPa)
        T02 = T2 * (1 + (self.gamma-1)/2 * M2**2)
        P02 = P2 * (1 + (self.gamma-1)/2 * M2**2)**(self.gamma/(self.gamma-1))  # Result in kPa
        
        # Calculate viscosity using Sutherland's law
        mu0 = 1.716e-5  # Reference viscosity at T0 = 273.15 K
        T0 = 273.15     # Reference temperature
        S = 110.4       # Sutherland constant for air
        mu2 = mu0 * (T2/T0)**(3/2) * (T0 + S)/(T2 + S)
        
        # Calculate Reynolds number based on half-length
        L = 0.034  # half-length in meters
        Re = rho2 * V2 * L / mu2
        
        # Calculate recovery temperature
        Pr = 0.7  # Prandtl number for air
        r = Pr**(1/2) if Re < 5e5 else Pr**(1/3)  # recovery factor
        Tr = T2 * (1 + r * (self.gamma-1)/2 * M2**2)
        
        return {
            'beta_deg': np.rad2deg(beta),
            'M2': M2,
            'P02': P02,      # Already in kPa
            'P2': P2,        # Already in kPa
            'T02': T02,
            'T2': T2,
            'V2': V2,
            'rho2': rho2,
            'mu2': mu2,
            'Re': Re,
            'Tr': Tr
        }

    def calculate_heat_transfer(self, results):
        """Calculate heat transfer properties for each Mach number"""
        k = 0.02  # Thermal conductivity [W/mK]
        Pr = 0.7  # Prandtl number for air
        L = 0.034  # half-length in meters
        heat_results = {}
        
        for M1, props in results.items():
            # Calculate Nusselt number for turbulent flow
            Re = props['Re']
            Nu = 0.0296 * (Re**(4/5)) * (Pr**(1/3))
            
            # Calculate heat transfer coefficient
            h = (Nu * k) / L  # [W/m²K]
            
            # Get recovery and surface temperatures
            Tr = props['Tr']  # Recovery temperature
            Ts = props['T2']  # Surface temperature (from TSP)
            
            # Calculate heat flux
            q = h * (Tr - Ts)  # [W/m²]
            
            # Store results with uncertainties
            dRe = props['uncertainties']['Re']
            dTr = props['uncertainties']['Tr']
            dTs = props['uncertainties']['T2']
            
            # Propagate uncertainties
            dNu = Nu * np.sqrt((4/5 * dRe/Re)**2)  # Simplified uncertainty propagation
            dh = (k/L) * dNu
            dq = np.sqrt((h * (Tr - Ts) * dh/h)**2 + (h * dTr)**2 + (h * dTs)**2)
            
            heat_results[M1] = {
                'Nu': Nu,
                'h': h,
                'q': q,
                'uncertainties': {
                    'Nu': dNu,
                    'h': dh,
                    'q': dq
                }
            }
        
        # Print results
        print("\nHeat Transfer Results:")
        print("="*100)
        print(f"{'Property':<20} {'M=2.0':<30} {'M=2.5':<30} {'M=3.0':<30}")
        print("-"*100)
        
        properties = [
            ('Nu', 'Nu'),
            ('h (W/m²K)', 'h'),
            ('q (W/m²)', 'q')
        ]
        
        for name, key in properties:
            print(f"{name:<20}", end=' ')
            for M1 in [2.0, 2.5, 3.0]:
                value = heat_results[M1][key]
                uncertainty = heat_results[M1]['uncertainties'][key]
                print(f"{value:>.4g} ± {uncertainty:>.4g}".ljust(30), end=' ')
            print()
        
        return heat_results

    def calculate_time_dependent_heat_flux(self, results, mach_num):
        """Calculate heat flux vs time for a specific Mach number run"""
        if isinstance(mach_num, float) and mach_num == 2.5:
            filename = f'Lab 4/F24 Lab 4 Data/Mach25.txt'
        else:
            filename = f'Lab 4/F24 Lab 4 Data/Mach{int(mach_num)}.txt'
        
        # Read temperature data
        data = pd.read_csv(filename, sep='\t',
                          names=['Time', 'Temperature', 'Stagnation', 'Static', 'Camera'])
        
        # Get heat transfer coefficient from previous calculations
        heat_results = self.calculate_heat_transfer(results)
        h = heat_results[mach_num]['h']
        Tr = results[mach_num]['Tr']
        
        # Calculate heat flux vs time
        time = data['Time']
        Ttc = data['Temperature']  # Thermocouple temperature
        q = h * (Tr - Ttc)
        
        return time, q

    def plot_nusselt_comparison(self, shock_results, heat_results):
        """Create plot comparing Nusselt numbers vs Reynolds numbers (Problem 8)"""
        plt.figure(figsize=(10, 6))
        
        Re_values = []
        Nu_values = []
        Nu_uncertainties = []
        mach_labels = []
        
        for M1 in [2.0, 2.5, 3.0]:
            Re = shock_results[M1]['Re']
            Nu = heat_results[M1]['Nu']
            dNu = heat_results[M1]['uncertainties']['Nu']
            
            Re_values.append(Re)
            Nu_values.append(Nu)
            Nu_uncertainties.append(dNu)
            mach_labels.append(f'M = {M1}')
        
        # Plot experimental points with error bars
        plt.errorbar(Re_values, Nu_values, yerr=Nu_uncertainties, 
                    fmt='o', capsize=5, label='Experimental Data')
        
        # Plot theoretical correlation line
        Re_line = np.linspace(min(Re_values)*0.8, max(Re_values)*1.2, 100)
        Nu_line = 0.0296 * Re_line**(4/5) * 0.7**(1/3)  # Pr = 0.7
        plt.plot(Re_line, Nu_line, 'r--', label='Theoretical (0.0296·Re⁴/⁵·Pr¹/³)')
        
        plt.xlabel('Reynolds Number')
        plt.ylabel('Nusselt Number')
        plt.title('Nusselt Number vs Reynolds Number Comparison')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        
        plt.savefig('Q8_nusselt_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_heat_flux_comparison(self, shock_results):
        """Create comparison plot of heat flux vs time for all Mach numbers (Problem 9)"""
        plt.figure(figsize=(12, 8))
        
        for M1 in [2.0, 2.5, 3.0]:
            time, q = self.calculate_time_dependent_heat_flux(shock_results, M1)
            plt.plot(time, q/1000, label=f'Mach {M1}')  # Convert to kW/m²
        
        plt.xlabel('Time (s)')
        plt.ylabel('Heat Flux (kW/m²)')
        plt.title('Heat Flux vs Time for Different Mach Numbers')
        plt.legend()
        plt.grid(True)
        
        # Add annotations explaining the trends
        plt.text(0.02, 0.98, 'Initial spike due to shock formation', 
                 transform=plt.gca().transAxes, verticalalignment='top')
        plt.text(0.02, 0.93, 'Steady-state values show increasing heat flux with Mach number',
                 transform=plt.gca().transAxes, verticalalignment='top')
        
        plt.savefig('Q9_heat_transfer.png', dpi=300, bbox_inches='tight')
        plt.close()

def process_shock_data(selected_temps):
    """Process all Mach numbers using selected temperatures"""
    calculator = ObliqueShockCalculator()
    mach_numbers = [2.0, 2.5, 3.0]
    results = {}
    
    for M1 in mach_numbers:
        # Get steady state pressures from data files
        P0, P1 = calculator.get_steady_state_pressures(M1)
        
        # Get corresponding temperature from selected points
        if M1 == 2.0:
            mach_key = 'M2'
        elif M1 == 2.5:
            mach_key = 'M2.5'
        else:
            mach_key = 'M3'
            
        T2 = selected_temps[mach_key][0] * 295  # Convert T/Tref back to T
        
        # Calculate properties with uncertainties
        results[M1] = calculator.calculate_shock_properties(M1, T2, P1)
        results[M1]['P0'] = P0  # Store stagnation pressure for reference
        
    # Print results table with uncertainties
    print("\nResults for 15° wedge:")
    print("="*100)
    print(f"{'Property':<20} {'M=2.0':<30} {'M=2.5':<30} {'M=3.0':<30}")
    print("-"*100)
    
    properties = [
        ('Shock angle β (deg)', 'beta_deg'),
        ('M2', 'M2'),
        ('P02 (kPa)', 'P02'),
        ('P2 (kPa)', 'P2'),
        ('T02 (K)', 'T02'),
        ('T2 (K)', 'T2'),
        ('V2 (m/s)', 'V2'),
        ('ρ2 (kg/m³)', 'rho2'),
        ('μ2 (Pa·s)', 'mu2'),
        ('Re', 'Re'),
        ('Tr (K)', 'Tr')
    ]
    
    for name, key in properties:
        print(f"{name:<20}", end=' ')
        for M1 in mach_numbers:
            if key in results[M1]['uncertainties']:
                value = results[M1][key]
                uncertainty = results[M1]['uncertainties'][key]
                print(f"{value:>.4g} ± {uncertainty:>.4g}".ljust(30), end=' ')
            else:
                print(f"{results[M1][key]:>.4g}".ljust(30), end=' ')
        print()
    
    return results

class TSPProcessor:
    def __init__(self):
        # Known TSP intensity ratios from data
        self.tsp_data = {
            'M2': {
                'Irat': [1/0.886361122988506, 1/0.864615162450593, 1/0.848534667311412],
                'dIrat': [0.0161464257935225, 0.0155414740595674, 0.0154517809727983]
            },
            'M2.5': {
                'Irat': [1/0.913784191525424, 1/0.888946055974843, 1/0.869854307898659],
                'dIrat': [0.0174466964703538, 0.0166358888116737, 0.0160636751344912]
            },
            'M3': {
                'Irat': [1/0.925674156172839, 1/0.900229311147186, 1/0.880659547985348],
                'dIrat': [0.0192029580085160, 0.0178914556010411, 0.0171101243435492]
            }
        }
        self.results = {}
        self.background_value = None
        self.shuttle_value = None
        self.threshold = None
        self.selected_temperatures = {}
        self.calculator = ObliqueShockCalculator()  # Initialize the calculator here
        
    def process_pressure_data(self, filename):
        """Process pressure and temperature data from text file"""
        data = pd.read_csv(filename, sep='\t',
                          names=['Time', 'Temperature', 'Stagnation', 'Static', 'Camera'])
        
        # Convert pressures to kPa
        P0 = 4137 * data['Stagnation'] + 101.325
        Pi = 1034 * data['Static'] + 101.325
        
        # Calculate Mach number
        M = np.sqrt(5 * ((P0/Pi)**(2/7) - 1))
        
        return data['Time'], data['Temperature'], P0, Pi, M
    
    def plot_pressure_temperature(self, mach_num):
        """Create pressure/temperature plot with Mach number (Problem 3)"""
        if isinstance(mach_num, float) and mach_num == 2.5:
            filename = f'Lab 4/F24 Lab 4 Data/Mach25.txt'
        else:
            filename = f'Lab 4/F24 Lab 4 Data/Mach{int(mach_num)}.txt'
            
        time, temp, P0, Pi, M = self.process_pressure_data(filename)
        
        # Store data for later use
        self.results[f'M{str(mach_num).replace(".", "")}'] = {
            'time': time,
            'temperature': temp,
            'P0': P0,
            'Pi': Pi,
            'M': M
        }
        
        # Create figure with three y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot pressures on primary y-axis
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Pressure (kPa)', color='black')
        l1 = ax1.plot(time, P0, 'b-', label='Stagnation Pressure')
        l2 = ax1.plot(time, Pi, 'r-', label='Static Pressure')
        ax1.tick_params(axis='y', labelcolor='black')
        
        # Create second y-axis for Temperature
        ax2 = ax1.twinx()
        ax2.spines['right'].set_position(('outward', 60))
        l3 = ax2.plot(time, temp, 'g-', label='Temperature')
        ax2.set_ylabel('Temperature (K)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Create third y-axis for Mach number
        ax3 = ax1.twinx()
        l4 = ax3.plot(time, M, 'k--', label='Mach Number')
        ax3.set_ylabel('Mach Number', color='k')
        ax3.tick_params(axis='y', labelcolor='k')
        
        # Combine legends
        lns = l1 + l2 + l3 + l4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper right')
        
        plt.title(f'Pressure, Temperature, and Mach Number vs Time (Mach {mach_num})')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(f'pressure_temp_M{str(mach_num).replace(".", "")}.png')
        plt.close()
        
    def get_peak_temperature(self, mach_num):
        """Get peak temperature for a given Mach number run"""
        if isinstance(mach_num, float) and mach_num == 2.5:
            filename = f'Lab 4/F24 Lab 4 Data/Mach25.txt'
        else:
            filename = f'Lab 4/F24 Lab 4 Data/Mach{int(mach_num)}.txt'
            
        # Read temperature data
        data = pd.read_csv(filename, sep='\t',
                          names=['Time', 'Temperature', 'Stagnation', 'Static', 'Camera'])
        
        # Get peak temperature
        return np.max(data['Temperature'])

    def get_points_for_image(self, image, filename):
        """Get threshold points for a specific image"""
        print(f"\nSelecting points for {filename}")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image, cmap='jet')
        plt.title('Click: 1) background point, 2) shuttle point')
        
        points = []
        background_value = [None]
        threshold = [None]
        
        def onclick(event):
            if event.inaxes == ax:
                x, y = int(event.xdata), int(event.ydata)
                points.append((y, x))
                
                if len(points) == 1:
                    background_value[0] = image[points[0]]
                    print(f"Background value: {background_value[0]}")
                elif len(points) == 2:
                    shuttle_value = image[points[1]]
                    print(f"Shuttle value: {shuttle_value}")
                    threshold[0] = (background_value[0] + shuttle_value) / 2
                    print(f"Threshold: {threshold[0]}")
                    plt.close()
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        plt.close('all')
        
        return threshold[0]

    def process_tsp_image(self, filename):
        """Process a single TSP data file into temperature map with Gaussian smoothing"""
        print(f"Loading {filename}...")
        
        # Load and reshape data
        data = np.loadtxt(filename, skiprows=4)
        values = data[:, 2]
        # Invert the intensity ratios
        values = 1/values
        image = values.reshape((800, 600)).T
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        # Apply Gaussian filter to reduce noise
        # kernel size must be odd, larger size = more smoothing
        kernel_size = (5, 5)  # You can adjust this value
        sigma = 2  # You can adjust this value
        smoothed_image = cv2.GaussianBlur(image, kernel_size, sigma)
        
        # Create mask based on threshold but invert it
        threshold = self.get_points_for_image(smoothed_image, filename)
        mask = smoothed_image < threshold  # Inverted mask logic
        
        # Convert to temperature map
        temp_map = 295 / smoothed_image  # Using smoothed image for temperature calculation
        masked_temp = np.ma.array(temp_map, mask=mask)
        
        return masked_temp

    def create_temperature_maps(self):
        """Create temperature maps for all runs"""
        print("Starting temperature map creation...")
        
        mach_numbers = ['M2', 'M2.5', 'M3']
        all_images = []
        
        # Process all images
        for mach in mach_numbers:
            for i in range(1, 4):
                filename = f'Lab 4/F24 Lab 4 Data/{mach}_{i}.dat'
                try:
                    masked_temp = self.process_tsp_image(filename)
                    all_images.append(masked_temp)
                    print(f"Successfully processed {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        print("\nCalculating global temperature range...")
        # Calculate global min/max for consistent color scale
        valid_temps = []
        for img in all_images:
            valid_temps.extend(img.compressed())
        vmin = np.percentile(valid_temps, 5)
        vmax = np.percentile(valid_temps, 95)
        print(f"Global temperature range: {vmin:.1f}K to {vmax:.1f}K")
        
        print("\nCreating final plot...")
        # Create plot
        fig = plt.figure(figsize=(15, 15))
        gs = GridSpec(3, 3, figure=fig)
        
        for idx, (mach, row) in enumerate(zip(mach_numbers, range(3))):
            for col in range(3):
                print(f"Plotting Mach {mach}, image {col+1}")
                ax = fig.add_subplot(gs[row, col])
                image_idx = idx * 3 + col
                im = ax.imshow(all_images[image_idx], 
                             vmin=vmin, vmax=vmax, 
                             cmap='jet')
                im.cmap.set_bad('white')  # Set masked values to white
                ax.axis('off')
                if col == 0:
                    ax.text(-0.2, 0.5, f'Mach {mach[1:]}', 
                           rotation=90, transform=ax.transAxes,
                           verticalalignment='center',
                           color='red')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Temperature [K]')
        
        print("\nSaving figure...")
        plt.savefig('temperature_maps_masked.png', bbox_inches='tight', dpi=300,
                    facecolor='white', edgecolor='none')
        print("Done!")
        plt.close('all')
        return all_images

    def select_temperatures(self, all_images):
        """Select temperature points from processed images and print T/Tref values"""
        mach_numbers = ['M2', 'M2.5', 'M3']
        T_ref = 295  # Reference temperature
        selected_temps = {}
        
        print("Select a point on each image to get temperature value...")
        
        for idx, mach in enumerate(mach_numbers):
            selected_temps[mach] = []
            for col in range(3):
                image_idx = idx * 3 + col
                
                # Create figure for point selection
                fig, ax = plt.subplots(figsize=(10, 10))
                im = ax.imshow(all_images[image_idx], cmap='jet')
                plt.colorbar(im, label='Temperature [K]')
                ax.set_title(f'Mach {mach[1:]} - Image {col+1}\nClick to select temperature point')
                
                point = {'temp': None}
                
                def onclick(event):
                    if event.inaxes == ax:
                        x, y = int(event.xdata), int(event.ydata)
                        temp = all_images[image_idx][y, x]
                        if not np.ma.is_masked(temp):
                            point['temp'] = temp
                            plt.close()
                
                fig.canvas.mpl_connect('button_press_event', onclick)
                plt.show()
                plt.close('all')
                
                if point['temp'] is not None:
                    temp = point['temp']
                    t_ratio = temp/T_ref
                    print(f"Mach {mach[1:]}, Image {col+1}:")
                    print(f"  Temperature: {temp:.2f} K")
                    print(f"  T/Tref: {t_ratio:.4f}\n")
                    selected_temps[mach].append(t_ratio)
                else:
                    print(f"Warning: No valid point selected for Mach {mach[1:]}, Image {col+1}")
        
        self.selected_temperatures = selected_temps
        return selected_temps
    
    def create_calibration_curve(self, selected_temps):
        """Create calibration curve using selected temperatures"""
        all_Irat = []
        all_Trat = []
        mach_markers = []
        
        # Create data points pairing each intensity ratio with its selected temperature
        for mach_key in ['M2', 'M2.5', 'M3']:
            for idx, (intensity, temp_ratio) in enumerate(zip(self.tsp_data[mach_key]['Irat'], 
                                                            selected_temps[mach_key])):
                # Store the intensity ratio directly (already in Iref/I form)
                I_ratio = intensity  # Remove the 1/intensity inversion
                
                all_Irat.append(I_ratio)
                all_Trat.append(temp_ratio)
                mach_markers.append(mach_key)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        markers = {'M2': 'o', 'M2.5': 's', 'M3': '^'}
        colors = {'M2': 'blue', 'M2.5': 'green', 'M3': 'red'}
        
        for mach in ['M2', 'M2.5', 'M3']:
            mask = [m == mach for m in mach_markers]
            plt.scatter(np.array(all_Trat)[mask], np.array(all_Irat)[mask],
                    marker=markers[mach], color=colors[mach],
                    label=f'Mach {mach[1:]}', s=100)
        
        # Linear fit
        z = np.polyfit(all_Trat, all_Irat, 1)
        p = np.poly1d(z)
        x_lin = np.linspace(min(all_Trat), max(all_Trat), 100)
        plt.plot(x_lin, p(x_lin), 'k-', 
                label=f'Linear Fit (Iref/I = {z[0]:.3f}×T/Tref + {z[1]:.3f})')
        
        plt.xlabel('T/Tref')
        plt.ylabel('Iref/I')  # Changed to Iref/I
        plt.title('TSP Calibration Curve')
        plt.legend()
        plt.grid(True)
        
        # Set axis limits
        plt.xlim(0.84, 0.92)
        plt.ylim(1.05, 1.20)  # Adjusted for inverted values
        
        plt.savefig('tsp_calibration_adjusted.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return z  # Return fit coefficients

    def process_all_data(self):
        """Complete processing workflow"""
        # First generate temperature maps and store the returned images
        all_images = self.create_temperature_maps()
        
        # Select temperature points from the maps
        selected_temps = self.select_temperatures(all_images)
        
        # Create new calibration curve with selected points
        self.create_calibration_curve(selected_temps)
        
        # Process shock data using selected temperatures
        shock_results = process_shock_data(selected_temps)
        
        # Calculate heat transfer and plot results
        heat_results = self.calculator.calculate_heat_transfer(shock_results)
        self.calculator.plot_nusselt_comparison(shock_results, heat_results)
        self.calculator.plot_heat_flux_comparison(shock_results)
        
        return shock_results, heat_results

def main():
    processor = TSPProcessor()

    print("Processing Problem 3...")
    for mach in [2.0, 2.5, 3.0]:
        processor.plot_pressure_temperature(mach)
    
    # Process all data and get results
    shock_results, heat_results = processor.process_all_data()

if __name__ == "__main__":
    main()