import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import cv2
import pandas as pd
from scipy.optimize import fsolve
import os

class UncertaintyCalculator:
    """Helper class for uncertainty calculations"""
    
    @staticmethod
    def sequential_perturbation(func, nominal_values, uncertainties, args=()):
        n = len(nominal_values)
        deltas = []
        
        # Calculate nominal value
        nominal = func(*nominal_values, *args)
        
        # Calculate perturbations
        for i in range(n):
            values_plus = nominal_values.copy()
            values_minus = nominal_values.copy()
            
            values_plus[i] += uncertainties[i]
            values_minus[i] -= uncertainties[i]
            
            result_plus = func(*values_plus, *args)
            result_minus = func(*values_minus, *args)
            
            delta = (result_plus - result_minus) / 2
            deltas.append(delta**2)
        
        # Total uncertainty
        total_uncertainty = np.sqrt(sum(deltas))
        
        return nominal, total_uncertainty

class Lab4Processor:
    def __init__(self):
        self.rect_coords = None
        self.image = None
        self.results = {
            'M2': {'rect': None, 'Irat': [], 'dIrat': []},
            'M2.5': {'rect': None, 'Irat': [], 'dIrat': []},
            'M3': {'rect': None, 'Irat': [], 'dIrat': []}
        }
        self.model_length = 0.068  # meters
        self.model_length_uncertainty = 0.0005  # meters
        self.calibration_curve = None
        self.uncertainty_calc = UncertaintyCalculator()
        
        # Instrument uncertainties
        self.pressure_uncertainty = {
            'stagnation': 4137 * 0.001,  # 1mV uncertainty in voltage measurement
            'static': 1034 * 0.001
        }
        self.temperature_uncertainty = 0.5  # K
        self.tsp_intensity_uncertainty = 0.01  # 1% uncertainty in intensity ratio

        # File mappings
        self.data_dir = "Lab 4/F24 Lab 4 Data"
        self.mach_files = {
            2.0: "Mach2.txt",
            2.5: "Mach25.txt",
            3.0: "Mach3.txt"
        }
        self.tsp_files = {
            2.0: ["M2_1.dat", "M2_2.dat", "M2_3.dat"],
            2.5: ["M2.5_1.dat", "M2.5_2.dat", "M2.5_3.dat"],
            3.0: ["M3_1.dat", "M3_2.dat", "M3_3.dat"]
        }

            # Store the known Irat and dIrat values
        self.tsp_data = {
            'M2': {
                'Irat': [0.886361122988506, 0.864615162450593, 0.848534667311412],
                'dIrat': [0.0161464257935225, 0.0155414740595674, 0.0154517809727983]
            },
            'M2.5': {
                'Irat': [0.913784191525424, 0.888946055974843, 0.869854307898659],
                'dIrat': [0.0174466964703538, 0.0166358888116737, 0.0160636751344912]
            },
            'M3': {
                'Irat': [0.925674156172839, 0.900229311147186, 0.880659547985348],
                'dIrat': [0.0192029580085160, 0.0178914556010411, 0.0171101243435492]
            }
        }

    def calculate_shock_properties(self, M1, P1, T1):
        """Calculate all shock properties for given conditions"""
        # Wedge angle (15 degrees)
        theta = np.radians(15)
        
        # Calculate shock angle
        beta = self.theta_beta_mach(M1, theta)
        
        # Get properties behind shock
        props = self.shock_properties(M1, beta, P1, T1)
        
        # Calculate viscosity
        mu2 = self.sutherland_viscosity(props['T2'])
        
        # Calculate Reynolds number
        Re = self.calculate_reynolds(props['V2'], self.model_length/2, props['rho2'], mu2)
        
        # Calculate recovery temperature
        Tr = self.calculate_recovery_temp(props['T2'], M1)
        
        # Store all properties
        props.update({
            'beta': np.degrees(beta),
            'mu': mu2,
            'Re': Re,
            'Tr': Tr
        })
        
        return props
    
    def process_run_data(self, mach_num):
        """Process data for a single Mach number run"""
        # Load pressure/temperature data
        if isinstance(mach_num, float) and mach_num == 2.5:
            filename = f'Lab 4/F24 Lab 4 Data/Mach25.txt'
        else:
            filename = f'Lab 4/F24 Lab 4 Data/Mach{int(mach_num)}.txt'
        time, temp, P0, Pi, M = self.process_pressure_data(filename)
        
        # Calculate mean values during steady state
        # Assuming steady state is in the middle third of the run
        start_idx = len(time) // 3
        end_idx = 2 * len(time) // 3
        
        M_mean = np.mean(M[start_idx:end_idx])
        P_mean = np.mean(Pi[start_idx:end_idx])
        T_mean = np.mean(temp[start_idx:end_idx])
        
        # Calculate shock properties
        props = self.calculate_shock_properties(M_mean, P_mean, T_mean)
        
        # Calculate heat transfer properties
        Nu = self.calculate_nusselt(props['Re'])
        h, q = self.calculate_heat_transfer(Nu, 0.02, self.model_length, 
                                          props['Tr'], T_mean)
        
        # Add to properties
        props.update({
            'Nu': Nu,
            'h': h,
            'q': q,
            'T_surface': T_mean
        })
        
        # Store results
        mach_key = f'M{str(mach_num).replace(".", "")}'
        self.results[mach_key]['shock_props'] = props
        
        return props
    
    def generate_results_table(self):
        """Generate table of results for all Mach numbers"""
        data = []
        columns = ['Mach', 'P2 (kPa)', 'T2 (K)', 'V2 (m/s)', 'rho2 (kg/m³)',
                  'mu (kg/m-s)', 'Re', 'Tr (K)', 'Nu', 'h (W/m²-K)', 'q (W/m²)']
        
        for mach_key, results in self.results.items():
            if mach_key.startswith('M') and 'shock_props' in results:
                props = results['shock_props']
                if props is not None:
                    # Convert mach key to number
                    if mach_key == 'M2.5':
                        mach_num = 2.5
                    else:
                        mach_num = float(mach_key[1:])
                    
                    data.append([
                        mach_num,
                        props['P2'], props['T2'], props['V2'], props['rho2'],
                        props['mu'], props['Re'], props['Tr'], props['Nu'],
                        props['h'], props['q']
                    ])
        
        df = pd.DataFrame(data, columns=columns)
        return df
    
    def plot_heat_transfer(self):
        """Create heat transfer plots for all runs"""
        plt.figure(figsize=(12, 8))
        
        for mach_key, results in self.results.items():
            if mach_key.startswith('M') and 'shock_props' in results:
                props = results['shock_props']
                if props is not None:
                    # Convert mach key to number
                    if mach_key == 'M2.5':
                        mach_num = 2.5
                    else:
                        mach_num = float(mach_key[1:])
                    
                    # Load temperature data
                    if mach_num == 2.5:
                        filename = f'Lab 4/F24 Lab 4 Data/Mach25.txt'
                    else:
                        filename = f'Lab 4/F24 Lab 4 Data/Mach{int(mach_num)}.txt'
                    time, temp, _, _, _ = self.process_pressure_data(filename)
                    
                    # Calculate heat transfer over time
                    q = props['h'] * (props['Tr'] - temp)
                    
                    plt.plot(time, q, label=f'Mach {mach_num}')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Heat Flux (W/m²)')
        plt.title('Heat Transfer Rate vs Time')
        plt.legend()
        plt.grid(True)
        plt.savefig('heat_transfer.png')
        plt.close()
        
    def process_pressure_data(self, filename):
        """Process pressure and temperature data"""
        data = pd.read_csv(filename, sep='\t', 
                          names=['Time', 'Temperature', 'Stagnation', 'Static', 'Camera'])
        
        # Convert pressures to kPa
        P0 = 4137 * data['Stagnation'] + 101.325
        Pi = 1034 * data['Static'] + 101.325
        
        # Add safety checks for pressure ratio
        ratio = P0/Pi
        valid_mask = (ratio >= 1.0) & (np.isfinite(ratio))  # Only calculate M where ratio is valid
        
        # Initialize M array with NaN
        M = np.full_like(P0, np.nan)
        
        # Calculate M only for valid indices
        M[valid_mask] = np.sqrt(5 * ((ratio[valid_mask])**(2/7) - 1))
        
        return data['Time'], data['Temperature'], P0, Pi, M
    
    # For Problem 3 - Modified plot_pressure_temperature method
    def plot_pressure_temperature(self, mach_num):
        """Create pressure/temperature plot with Mach number (Question 3)"""
        if isinstance(mach_num, float) and mach_num == 2.5:
            filename = f'Lab 4/F24 Lab 4 Data/Mach25.txt'
        else:
            filename = f'Lab 4/F24 Lab 4 Data/Mach{int(mach_num)}.txt'
        time, temp, P0, Pi, M = self.process_pressure_data(filename)
        
        # Create figure with single plot and three y-axes
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
        
        plt.savefig(f'pressure_temp_M{mach_num}.png')
        plt.close()
        
    def process_tsp_image(self, filename, show_plot=True):
        """Process a single TSP image"""
        data = np.loadtxt(filename, skiprows=4)
        values = data[:, 2]
        B = values.reshape((800, 600)).T
        self.image = cv2.rotate(B, cv2.ROTATE_90_CLOCKWISE)
        
        if show_plot:
            # Create figure for rectangle selection
            fig, ax = plt.subplots()
            im = ax.imshow(self.image, cmap='jet')
            plt.colorbar(im, label='I/I0')
            plt.title(f'Select region for {filename}\nClose window when done')
            
            # Initialize rectangle coordinates
            self.rect_coords = None
            
            def onselect(eclick, erelease):
                """Callback for rectangle selection"""
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata
                self.rect_coords = (min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1))
            
            # Create rectangle selector
            rs = RectangleSelector(ax, onselect, interactive=True)
            plt.show()
            
            if self.rect_coords is None:
                raise ValueError("No region selected")
        else:
            # Use previously stored rectangle coordinates for this Mach number
            # Extract Mach key from filename
            if '2.5' in filename or 'M25' in filename:
                mach_key = 'M2.5'
            elif 'M2' in filename:
                mach_key = 'M2'
            elif 'M3' in filename:
                mach_key = 'M3'
            else:
                raise ValueError(f"Could not extract Mach number from filename: {filename}")
            
            self.rect_coords = self.results[mach_key]['rect']
            if self.rect_coords is None:
                raise ValueError(f"No stored rectangle coordinates for {mach_key}")
        
        # Calculate stats for selected region
        x, y, w, h = [int(v) for v in self.rect_coords]
        roi = self.image[y:y+h, x:x+w]
        
        Irat = np.mean(roi)
        dIrat = np.mean(np.std(roi, axis=0))
        
        # Extract Mach key from filename (M2, M2.5, or M3)
        if '2.5' in filename or 'M25' in filename:
            mach_key = 'M2.5'
        elif 'M2' in filename:
            mach_key = 'M2'
        elif 'M3' in filename:
            mach_key = 'M3'
        else:
            raise ValueError(f"Could not extract Mach number from filename: {filename}")
        
        # Store rectangle coordinates for later use
        if self.results[mach_key]['rect'] is None:
            self.results[mach_key]['rect'] = (x, y, w, h)
        
        return Irat, dIrat
    
    # Modified process_all_tsp_images to use stored values
    def process_all_tsp_images(self):
        """Process TSP images using stored values"""
        for mach_key in ['M2', 'M2.5', 'M3']:
            # Use the stored values directly
            self.results[mach_key]['Irat'] = self.tsp_data[mach_key]['Irat']
            self.results[mach_key]['dIrat'] = self.tsp_data[mach_key]['dIrat']
    
    def create_temperature_maps(self, calibration_curve):
        """Create temperature maps (Question 5)"""
        # Create figure with 3x3 subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        vmin, vmax = None, None  # Will store global min/max temperatures
        
        # First pass to get global min/max
        for mach, files in zip(['M2', 'M2.5', 'M3'], axes):
            for i, file in enumerate([f'Lab 4/F24 Lab 4 Data/{mach}_{j}.dat' for j in range(1, 4)]):
                data = np.loadtxt(file, skiprows=4)
                values = data[:, 2]
                image = values.reshape((800, 600)).T
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                
                # Convert I/I0 to temperature using calibration curve
                temp_map = calibration_curve(image)  # You'll need to implement this
                
                if vmin is None:
                    vmin, vmax = np.min(temp_map), np.max(temp_map)
                else:
                    vmin = min(vmin, np.min(temp_map))
                    vmax = max(vmax, np.max(temp_map))
        
        # Second pass to plot with consistent colorbar
        for i, (mach, row) in enumerate(zip(['M2', 'M2.5', 'M3'], axes)):
            for j, file in enumerate([f'Lab 4/F24 Lab 4 Data/{mach}_{k}.dat' for k in range(1, 4)]):
                data = np.loadtxt(file, skiprows=4)
                values = data[:, 2]
                image = values.reshape((800, 600)).T
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                
                temp_map = calibration_curve(image)
                im = row[j].imshow(temp_map, vmin=vmin, vmax=vmax, cmap='jet')
                row[j].axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Temperature [K]')
        
        plt.savefig('temperature_maps.png', bbox_inches='tight')
        plt.close()

            
    # Add helper methods from the shock calculations script
    def theta_beta_mach(self, M1, theta):
        """θ-β-M relation solver"""
        gamma = 1.4
        
        def equation(beta):
            return np.tan(theta) - 2 * (1/np.tan(beta)) * \
                   ((M1**2 * np.sin(beta)**2 - 1)/(M1**2 * (gamma + np.cos(2*beta)) + 2))
        
        beta_guess = np.radians(45)
        beta = fsolve(equation, beta_guess)[0]
        return beta
    
    def shock_properties(self, M1, beta, P1=101.325, T1=295):
        """Calculate properties behind oblique shock"""
        gamma = 1.4
        R = 287
        
        # Add safety checks
        if not np.isfinite(M1) or M1 <= 0:
            return {
                'P2': np.nan,
                'T2': np.nan,
                'rho2': np.nan,
                'V2': np.nan
            }
        
        Mn1 = M1 * np.sin(beta)
        
        # Check for valid shock conditions
        if Mn1 <= 1:
            return {
                'P2': np.nan,
                'T2': np.nan,
                'rho2': np.nan,
                'V2': np.nan
            }
        
        P2_P1 = (2*gamma*Mn1**2 - (gamma-1))/(gamma+1)
        T2_T1 = (2*gamma*Mn1**2 - (gamma-1))*((gamma-1)*Mn1**2 + 2)/((gamma+1)**2*Mn1**2)
        rho2_rho1 = (gamma+1)*Mn1**2/((gamma-1)*Mn1**2 + 2)
        
        P2 = P2_P1 * P1
        T2 = T2_T1 * T1
        rho2 = rho2_rho1 * (P1/(R*T1))
        
        V1 = M1 * np.sqrt(gamma * R * T1)
        
        # Add safety check for V2 calculation
        denominator = 2*gamma*Mn1**2 - (gamma-1)
        if denominator <= 0:
            V2 = np.nan
        else:
            V2 = V1 * np.sqrt(((gamma-1)*Mn1**2 + 2)/denominator)
        
        return {
            'P2': P2,
            'T2': T2,
            'rho2': rho2,
            'V2': V2
        }
    
    @staticmethod
    def sutherland_viscosity(T):
        """Sutherland's law for viscosity"""
        mu0 = 1.716e-5
        T0 = 273.15
        S = 110.4
        return mu0 * (T/T0)**(3/2) * (T0 + S)/(T + S)
    
    @staticmethod
    def calculate_reynolds(V, L, rho, mu):
        """Reynolds number calculation"""
        return rho * V * L / mu
    
    @staticmethod
    def calculate_recovery_temp(T, M, Pr=0.7, r=0.896):
        """Recovery temperature calculation"""
        gamma = 1.4
        return T * (1 + r * (gamma-1)/2 * M**2)
    
    @staticmethod
    def calculate_nusselt(Re, Pr=0.7):
        """Nusselt number calculation"""
        return 0.0296 * Re**(4/5) * Pr**(1/3)
    
    @staticmethod
    def calculate_heat_transfer(Nu, k, L, Tr, Ts):
        """Heat transfer coefficient and flux calculation"""
        h = Nu * k / L
        q = h * (Tr - Ts)
        return h, q
    
            
    def process_question_3(self):
        """
        Q3: Process and plot pressure/temperature data
        """
        mach_numbers = [2, 2.5, 3]
        self.results['Q3'] = {}
        
        for mach in mach_numbers:
            # Construct filename with proper format
            if mach == 2.5:
                filename = 'Lab 4/F24 Lab 4 Data/Mach25.txt'
            else:
                filename = f'Lab 4/F24 Lab 4 Data/Mach{int(mach)}.txt'
            
            data = pd.read_csv(filename, sep='\t',
                             names=['Time', 'Temperature', 'Stagnation', 'Static', 'Camera'])
            
            # Convert voltages to pressures (kPa)
            P0 = 4137 * data['Stagnation'] + 101.325
            Pi = 1034 * data['Static'] + 101.325
            
            # Calculate Mach number
            M = np.sqrt(5 * ((P0/Pi)**(2/7) - 1))
            
            # Store results
            self.results['Q3'][mach] = {
                'time': data['Time'],
                'temperature': data['Temperature'],
                'P0': P0,
                'Pi': Pi,
                'M': M
            }
            
            # Calculate uncertainties
            P0_uncertainty = np.full_like(P0, self.pressure_uncertainty['stagnation'])
            Pi_uncertainty = np.full_like(Pi, self.pressure_uncertainty['static'])
            
            M_nominal = np.zeros_like(P0)
            M_uncertainty = np.zeros_like(P0)
            
            for i in range(len(P0)):
                M_nominal[i], M_uncertainty[i] = self.calculate_mach_uncertainty(
                    P0[i], Pi[i])
            
            self.results['Q3'][mach].update({
                'P0_uncertainty': P0_uncertainty,
                'Pi_uncertainty': Pi_uncertainty,
                'M_uncertainty': M_uncertainty
            })
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            ax1.plot(data['Time'], P0, 'b-', label='Stagnation Pressure')
            ax1.plot(data['Time'], Pi, 'r-', label='Static Pressure')
            ax1.set_ylabel('Pressure (kPa)')
            ax1.legend()
            ax1.grid(True)
            
            ax2.plot(data['Time'], data['Temperature'], 'g-', label='Temperature')
            ax2.set_ylabel('Temperature (K)')
            
            ax3 = ax2.twinx()
            ax3.plot(data['Time'], M, 'k--', label='Mach Number')
            ax3.set_ylabel('Mach Number')
            
            plt.xlabel('Time (s)')
            plt.savefig(f'Q3_M{str(mach).replace(".", "")}.png')
            plt.close()
            
    def process_question_4(self):
        """Process TSP calibration data and create calibration curve plot"""
        # Lists to store all data points
        all_Irat = []
        all_Trat = []
        
        # Process data for each Mach number
        for mach_key in ['M2', 'M2.5', 'M3']:
            # Get the temperature data directly from the data files
            if mach_key == 'M2.5':
                filename = 'Lab 4/F24 Lab 4 Data/Mach25.txt'
            else:
                mach_num = float(mach_key.replace('M', ''))
                filename = f'Lab 4/F24 Lab 4 Data/Mach{int(mach_num)}.txt'
                
            # Load and process temperature data
            data = pd.read_csv(filename, sep='\t',
                              names=['Time', 'Temperature', 'Stagnation', 'Static', 'Camera'])
            
            # Get the reference temperature from the start of the run
            T_ref = data['Temperature'].iloc[0]  # Use initial temperature as reference
            
            # Get the steady state temperature (middle third of the run)
            temp_data = data['Temperature']
            steady_start = len(temp_data) // 3
            steady_end = 2 * len(temp_data) // 3
            T = np.mean(temp_data[steady_start:steady_end])
            
            # Get the stored Irat values
            Irat_values = self.tsp_data[mach_key]['Irat']
            
            # Calculate T/Tref for each Irat
            for Irat in Irat_values:
                all_Irat.append(Irat)
                all_Trat.append(T/T_ref)
        
        # Create the calibration plot
        plt.figure(figsize=(8, 6))
        plt.scatter(all_Trat, all_Irat, color='blue', label='Experimental Data')
        
        # Fit a linear regression
        z = np.polyfit(all_Trat, all_Irat, 1)
        p = np.poly1d(z)
        x_lin = np.linspace(min(all_Trat), max(all_Trat), 100)
        plt.plot(x_lin, p(x_lin), 'r-', label=f'Linear Fit (y = {z[0]:.3f}x + {z[1]:.3f})')
        
        plt.xlabel('T/Tref')
        plt.ylabel('I/Iref')
        plt.title('TSP Calibration Curve')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('tsp_calibration.png')
        plt.close()
        
        # Store calibration curve for later use
        self.calibration_curve = lambda x: (x - z[1]) / z[0]  # Inverse function to get T from I
    
    def process_question_5(self):
        """
        Q5: Create temperature maps using calibration
        """
        # Create calibration curve using stored tsp_data values
        def calibration_curve(Irat):
            """Simple linear calibration curve using stored values"""
            # Using stored values to create calibration
            # You may want to adjust this calibration relationship
            return 1.0 / Irat  # Example conversion - adjust based on your needs
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        vmin, vmax = None, None
        
        # First pass to get global min/max
        for i, mach in enumerate(['M2', 'M2.5', 'M3']):
            for j in range(3):
                filename = f'Lab 4/F24 Lab 4 Data/{mach}_{j+1}.dat'
                if mach == 'M2.5':
                    filename = f'Lab 4/F24 Lab 4 Data/M2.5_{j+1}.dat'
                    
                data = np.loadtxt(filename, skiprows=4)
                values = data[:, 2]
                image = values.reshape((800, 600)).T
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                
                temp_map = calibration_curve(image)
                
                if vmin is None:
                    vmin, vmax = np.min(temp_map), np.max(temp_map)
                else:
                    vmin = min(vmin, np.min(temp_map))
                    vmax = max(vmax, np.max(temp_map))
        
        # Second pass to plot with consistent colorbar
        for i, mach in enumerate(['M2', 'M2.5', 'M3']):
            for j in range(3):
                filename = f'Lab 4/F24 Lab 4 Data/{mach}_{j+1}.dat'
                if mach == 'M2.5':
                    filename = f'Lab 4/F24 Lab 4 Data/M2.5_{j+1}.dat'
                    
                data = np.loadtxt(filename, skiprows=4)
                values = data[:, 2]
                image = values.reshape((800, 600)).T
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                
                temp_map = calibration_curve(image)
                im = axes[i, j].imshow(temp_map, vmin=vmin, vmax=vmax, cmap='jet')
                axes[i, j].set_title(f'{mach} Run {j+1}')
                axes[i, j].axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Temperature [K]')
        
        # Make sure results directory exists
        os.makedirs('Lab 4/results', exist_ok=True)
        plt.savefig('Lab 4/results/temperature_maps.png', bbox_inches='tight')
        plt.close()
        
    def process_question_7(self):
        """Calculate shock properties with uncertainties"""
        self.results['Q7'] = {}
        
        for mach in [2.0, 2.5, 3.0]:
            # Load data directly from file
            if mach == 2.5:
                filename = 'Lab 4/F24 Lab 4 Data/Mach25.txt'
            else:
                filename = f'Lab 4/F24 Lab 4 Data/Mach{int(mach)}.txt'
                
            # Get the data
            time, temp, P0, Pi, M = self.process_pressure_data(filename)
            
            # Get steady state values (middle third of the run)
            steady_start = len(time) // 3
            steady_end = 2 * len(time) // 3
            
            M1 = np.nanmean(M[steady_start:steady_end])
            P1 = np.nanmean(Pi[steady_start:steady_end])
            T1 = np.nanmean(temp[steady_start:steady_end])
            
            # Check for valid data
            if np.isfinite(M1) and np.isfinite(P1) and np.isfinite(T1):
                # Calculate properties with uncertainties
                props, uncertainties = self.calculate_shock_properties_with_uncertainty(
                    M1, P1, T1)
                
                # Add Reynolds number calculation
                if np.isfinite(props['V2']) and np.isfinite(props['rho2']):
                    mu = self.sutherland_viscosity(props['T2'])
                    Re = self.calculate_reynolds(props['V2'], self.model_length/2, 
                                              props['rho2'], mu)
                    props['Re'] = Re
                else:
                    props['Re'] = np.nan
                    
                self.results['Q7'][mach] = {
                    'properties': props,
                    'uncertainties': uncertainties
                }
            else:
                self.results['Q7'][mach] = {
                    'properties': {'Re': np.nan},
                    'uncertainties': {}
                }
        
    def process_question_8(self):
        """
        Q8: Calculate heat transfer coefficient
        """
        if 'Q7' not in self.results:
            raise ValueError("Must run process_question_7 first!")
            
        self.results['Q8'] = {}
        k = 0.02  # W/m-K
        
        for mach in [2.0, 2.5, 3.0]:
            q7_data = self.results['Q7'][mach]
            
            # Access Re from the properties dictionary
            Re = q7_data['properties']['Re']
            
            # Only calculate if Re is valid
            if np.isfinite(Re):
                # Calculate Nusselt number
                Nu = self.calculate_nusselt(Re)
                
                # Calculate heat transfer coefficient
                h = Nu * k / self.model_length
            else:
                Nu = np.nan
                h = np.nan
            
            self.results['Q8'][mach] = {
                'Nu': Nu,
                'h': h
            }
            
    def process_question_9(self):
        """
        Q9: Calculate heat transfer rate vs time
        """
        if 'Q8' not in self.results:
            raise ValueError("Must run process_question_8 first!")
            
        plt.figure(figsize=(10, 6))
        
        for mach in [2.0, 2.5, 3.0]:
            # Load data directly from file
            if mach == 2.5:
                filename = 'Lab 4/F24 Lab 4 Data/Mach25.txt'
            else:
                filename = f'Lab 4/F24 Lab 4 Data/Mach{int(mach)}.txt'
                
            # Get the data
            time, temp, _, _, _ = self.process_pressure_data(filename)
            
            # Get heat transfer coefficient from Q8
            h = self.results['Q8'][mach]['h']
            
            # Get recovery temperature from Q7
            Tr = self.results['Q7'][mach]['properties'].get('T2', np.nan)  # Using T2 as recovery temperature
            
            if np.isfinite(h) and np.isfinite(Tr):
                # Calculate q(t)
                q = h * (Tr - temp)
                plt.plot(time, q, label=f'Mach {mach}')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Heat Flux (W/m²)')
        plt.title('Heat Transfer Rate vs Time')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('Q9_heat_transfer.png')
        plt.close()

    def calculate_mach_uncertainty(self, P0, Pi):
        """Calculate Mach number and its uncertainty"""
        def mach_func(P0, Pi):
            return np.sqrt(5 * ((P0/Pi)**(2/7) - 1))
        
        nominal = [P0, Pi]
        uncertainties = [self.pressure_uncertainty['stagnation'],
                        self.pressure_uncertainty['static']]
        
        return self.uncertainty_calc.sequential_perturbation(
            mach_func, nominal, uncertainties)
    
    
    def calculate_shock_properties_with_uncertainty(self, M1, P1, T1):
        """Calculate shock properties with uncertainties"""
        def shock_property_func(M, P, T):
            beta = self.theta_beta_mach(M, np.radians(15))
            props = self.shock_properties(M, beta, P, T)
            return props
        
        # Nominal values and uncertainties
        nominal = [M1, P1, T1]
        uncertainties = [
            0.05,  # Estimated Mach number uncertainty
            self.pressure_uncertainty['static'],
            self.temperature_uncertainty
        ]
        
        # Calculate properties and uncertainties
        props_nominal = shock_property_func(*nominal)
        props_uncertainties = {}
        
        for key in props_nominal:
            def extract_func(*args):
                return shock_property_func(*args)[key]
            
            _, uncertainty = self.uncertainty_calc.sequential_perturbation(
                extract_func, nominal, uncertainties)
            props_uncertainties[f'{key}_uncertainty'] = uncertainty
        
        return props_nominal, props_uncertainties
    
    
    def save_shock_properties_table(self):
        """Save shock properties with uncertainties to CSV"""
        data = []
        columns = ['Mach', 'P2 (kPa)', 'T2 (K)', 'V2 (m/s)', 'rho2 (kg/m³)',
                  'mu (kg/m-s)', 'Re', 'Tr (K)']
        
        for mach in [2.0, 2.5, 3.0]:
            row = [mach]  # Initialize row with Mach number
            props = self.results['Q7'][mach]['properties']
            uncertainties = self.results['Q7'][mach]['uncertainties']
            
            # Calculate mu if it doesn't exist but T2 does
            if 'mu' not in props and 'T2' in props:
                props['mu'] = self.sutherland_viscosity(props['T2'])
            
            # Calculate Tr if it doesn't exist but T2 does
            if 'Tr' not in props and 'T2' in props:
                props['Tr'] = self.calculate_recovery_temp(props['T2'], mach)
            
            # Add each property with its uncertainty, handling missing values
            for key in ['P2', 'T2', 'V2', 'rho2', 'mu', 'Re', 'Tr']:
                value = props.get(key, np.nan)
                uncertainty = uncertainties.get(f'{key}_uncertainty', np.nan)
                
                if np.isfinite(value) and np.isfinite(uncertainty):
                    row.append(f'{value:.2f} ± {uncertainty:.2f}')
                else:
                    row.append('N/A')
            
            data.append(row)
        
        df = pd.DataFrame(data, columns=columns)
        df.to_csv('shock_properties_with_uncertainties.csv', index=False)


    def run_full_analysis(self):
        
        # Run analysis in correct order
        print("Processing Question 3...")
        self.plot_pressure_temperature(2.0)
        self.plot_pressure_temperature(2.5)
        self.plot_pressure_temperature(3.0)
        
        print("Processing Question 4...")
        self.process_question_4()
        
        print("Processing Question 5...")
        self.process_question_5()
        
        print("Processing Question 7...")
        self.process_question_7()
        
        print("Processing Question 8...")
        self.process_question_8()
        
        print("Processing Question 9...")
        self.process_question_9()
        
        print("Saving shock properties table...")
        self.save_shock_properties_table()
        
        print("Analysis complete!")



if __name__ == "__main__":
    processor = Lab4Processor()
    processor.run_full_analysis()