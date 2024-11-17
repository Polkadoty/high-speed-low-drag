import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, curve_fit
from pathlib import Path
import os

class PressureAnalysis:
    def __init__(self):
        self.gamma = 1.4
        self.p_atm = 14.7  # psi
        self.R = 287.05    # Gas constant for air [J/kg·K]
        self.T0 = 297      # Reference temperature [K]
        
    def read_data_file(self, file_path):
        """Read pressure data from file"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        data_start = next(i for i, line in enumerate(lines) if "X_Value" in line)
        data = np.genfromtxt(lines[data_start+1:], delimiter='\t', usecols=(1, 2, 3))
        return data
    
    def voltage_to_pressure(self, voltage, conversion_factor):
        """Convert voltage to pressure"""
        return voltage * (conversion_factor / 0.1)
    
    def find_steady_state(self, data, window_size=50):
        """Find steady state region in data"""
        peak_index = np.argmax(np.abs(data))
        search_start = max(0, peak_index - window_size*2)
        search_end = min(len(data), peak_index + window_size*2)
        
        std_dev = np.array([np.std(data[i:i+window_size]) 
                           for i in range(search_start, search_end-window_size)])
        stable_start = search_start + np.argmin(std_dev)
        
        return stable_start, stable_start + window_size
    
    def calculate_mach_number(self, p0, p, gamma=1.4):
        """Calculate Mach number from pressure ratio"""
        if p0 <= p or p <= 0:
            return np.nan
        return np.sqrt((2 / (gamma - 1)) * ((p0 / p)**((gamma - 1) / gamma) - 1))
    
    def calculate_reynolds_numbers(self, mach_number, pressure, temperature, diameter):
        """Calculate Reynolds numbers"""
        # Sutherland's law constants
        C = 120  # K
        T0 = 291.15  # K
        mu0 = 1.827e-5  # Pa·s

        T_ratio = 1 + (self.gamma - 1) / 2 * mach_number**2
        T = temperature * T_ratio
        rho = pressure / (self.R * T)
        V = mach_number * np.sqrt(self.gamma * self.R * T)
        mu = mu0 * (T / T0)**(3/2) * (T0 + C) / (T + C)

        Re_unit = rho * V / mu
        Re_D = Re_unit * diameter

        return Re_unit, Re_D

    def process_file(self, file_path):
        """Process a single data file"""
        data = self.read_data_file(file_path)
        
        # Get filename as run identifier
        run_name = Path(file_path).stem
        
        # Convert voltages to pressures
        p0 = self.voltage_to_pressure(data[:, 0], 60)
        p1 = self.voltage_to_pressure(data[:, 1], 15)
        p2 = self.voltage_to_pressure(data[:, 2], 15)
        
        # Find steady state
        start, end = self.find_steady_state(p0)
        
        # Calculate averages and uncertainties
        avg_p0 = np.mean(p0[start:end])
        avg_p1 = np.mean(p1[start:end])
        avg_p2 = np.mean(p2[start:end])
        
        dp0 = 0.01 * avg_p0
        dp1 = 0.01 * avg_p1
        dp2 = 0.01 * avg_p2
        
        # Calculate Mach number using p0/p1 ratio (p1 is freestream static pressure)
        mach = self.calculate_mach_number(avg_p0 + self.p_atm, avg_p1 + self.p_atm)
        
        # Update uncertainty calculation for new pressure ratio
        dmach = mach * np.sqrt((dp0/(avg_p0 + self.p_atm))**2 + (dp1/(avg_p1 + self.p_atm))**2)
        
        # Calculate pressure ratio p2/p1 (surface to freestream)
        p_ratio = (avg_p2 + self.p_atm) / (avg_p1 + self.p_atm)
        
        # Calculate Reynolds numbers and uncertainty
        diameter = 0.0163  # m
        Re_unit, Re_D = self.calculate_reynolds_numbers(
            mach, (avg_p2 + self.p_atm) * 6894.75729, self.T0, diameter
        )
        
        # Simplified Reynolds number uncertainty (assuming Mach number dominates)
        dRe_D = Re_D * dmach/mach
        
        return {
            'run': run_name,
            'mach': mach,
            'mach_uncertainty': dmach,
            'p0': avg_p0,
            'p1': avg_p1,
            'p2': avg_p2,
            'p_ratio': p_ratio,
            'Re_unit': Re_unit,
            'Re_D': Re_D,
            'Re_D_uncertainty': dRe_D
        }

    def create_flow_conditions_table(self, results_df):
        """Create table of flow conditions and uncertainties"""
        flow_conditions = pd.DataFrame({
            'Run': results_df['run'],
            'Mach Number': results_df['mach'].round(3),
            'P0 (psi)': (results_df['p0'] + self.p_atm).round(2),
            'P1 (psi)': (results_df['p1'] + self.p_atm).round(2),
            'P2 (psi)': (results_df['p2'] + self.p_atm).round(2),
            'P2/P1': results_df['p_ratio'].round(3),
            'Re_D': results_df['Re_D'].round(0)
        })
        return flow_conditions

class TheoryCalculations:
    def __init__(self):
        self.gamma = 1.4
        
    def oblique_shock_angle(self, M1, theta):
        """Calculate oblique shock angle"""
        def shock_equation(beta):
            return np.tan(theta) - 2/np.tan(beta) * (M1**2*np.sin(beta)**2 - 1) / \
                   (M1**2*(self.gamma + np.cos(2*beta)) + 2)
        
        # Initial guess for weak shock solution
        beta_guess = theta + 0.1
        beta = fsolve(shock_equation, beta_guess)[0]
        return beta
    
    def pressure_ratio(self, M1, beta):
        """Calculate pressure ratio across oblique shock"""
        Mn1 = M1 * np.sin(beta)
        return 1 + 2*self.gamma/(self.gamma + 1) * (Mn1**2 - 1)
    
    def theoretical_curves(self, mach_range):
        """Generate theoretical curves for comparison"""
        theta = np.radians(10)  # 10° wedge angle
        results = []
        
        for M in mach_range:
            beta = self.oblique_shock_angle(M, theta)
            p_ratio = self.pressure_ratio(M, beta)
            results.append({
                'mach': M,
                'beta': np.degrees(beta),
                'p_ratio': p_ratio
            })
            
        return pd.DataFrame(results)

    def calculate_surface_pressure(self, mach, theta):
        """Calculate theoretical surface pressure for given Mach and deflection"""
        beta = self.oblique_shock_angle(mach, theta)
        p_ratio = self.pressure_ratio(mach, beta)
        return p_ratio, beta

    def analyze_wedge_flow(self, experimental_df, angle_measurements_df):
        """Analyze wedge flow and compare with theory"""
        results = []
        
        # Convert angle measurements to radians for calculations
        angle_measurements_df['shock_angle_rad'] = np.radians(angle_measurements_df['shock_angle'])
        angle_measurements_df['deflection_angle_rad'] = np.radians(angle_measurements_df['deflection_angle'])
        
        # Merge experimental data with angle measurements
        merged_df = experimental_df.merge(angle_measurements_df, on='run', how='inner')
        
        for _, row in merged_df.iterrows():
            # Calculate theoretical values
            theta = row['deflection_angle_rad']
            beta_theory = self.oblique_shock_angle(row['mach'], theta)
            p_ratio_theory = self.pressure_ratio(row['mach'], beta_theory)
            
            results.append({
                'run': row['run'],
                'mach': row['mach'],
                'shock_angle_measured': row['shock_angle'],
                'shock_angle_theory': np.degrees(beta_theory),
                'deflection_angle': row['deflection_angle'],
                'p_ratio_measured': row['p_ratio'],
                'p_ratio_theory': p_ratio_theory,
                'shock_angle_error': abs(row['shock_angle'] - np.degrees(beta_theory)),
                'p_ratio_error': abs(row['p_ratio'] - p_ratio_theory) / p_ratio_theory * 100
            })
        
        return pd.DataFrame(results)

class Visualization:
    def __init__(self, output_path="Lab 2/report2/figures"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        # Create tables directory
        tables_path = self.output_path.parent / "tables"
        tables_path.mkdir(parents=True, exist_ok=True)
        
    def plot_pressure_comparison(self, experimental, theoretical):
        """Plot pressure ratio comparison"""
        plt.figure(figsize=(10, 6))
        plt.plot(theoretical['mach'], theoretical['p_ratio'], 'k-', label='Theory')
        plt.scatter(experimental['mach'], experimental['p_ratio'], c='r', label='Experimental')
        plt.xlabel('Mach Number')
        plt.ylabel('Pressure Ratio (P2/P1)')
        plt.title('Pressure Ratio Comparison')
        plt.grid(True)
        plt.legend()
        plt.savefig(self.output_path / 'pressure_comparison.png', dpi=300)
        plt.close()
        
    def plot_shock_angle_comparison(self, experimental, theoretical, angle_measurements):
        """Plot shock angle comparison"""
        plt.figure(figsize=(10, 6))
        
        # Plot theoretical curve
        plt.plot(theoretical['mach'], theoretical['beta'], 'k-', label='Theory')
        
        # Merge experimental data with angle measurements and plot
        merged_data = experimental.merge(angle_measurements, on='run', how='inner')
        plt.scatter(merged_data['mach'], merged_data['shock_angle'], c='r', label='Experimental')
        
        plt.xlabel('Mach Number')
        plt.ylabel('Shock Angle (degrees)')
        plt.title('Shock Angle Comparison')
        plt.grid(True)
        plt.legend()
        plt.savefig(self.output_path / 'shock_angle_comparison.png', dpi=300)
        plt.close()
        
    def plot_reynolds_mach(self, data):
        """Plot Reynolds number vs Mach number"""
        plt.figure(figsize=(10, 6))
        
        # Check if uncertainty data is available
        if 'Re_D_uncertainty' in data.columns:
            plt.errorbar(data['mach'], data['Re_D'], 
                        yerr=data['Re_D_uncertainty'],
                        fmt='o', capsize=5, label='Re_D with uncertainty')
        else:
            plt.scatter(data['mach'], data['Re_D'], 
                       label='Re_D')
        
        plt.xlabel('Mach Number')
        plt.ylabel('Reynolds Number (Re_D)')
        plt.title('Reynolds Number vs Mach Number')
        plt.grid(True)
        plt.legend()
        plt.savefig(self.output_path / 'reynolds_mach.png', dpi=300)
        plt.close()

    def create_latex_tables(self, flow_conditions, wedge_analysis):
        """Create LaTeX tables for the report"""
        # Flow conditions table
        flow_table = flow_conditions.to_latex(index=False, float_format="%.3f")
        with open(self.output_path.parent / "tables/flow_conditions.tex", "w") as f:
            f.write(flow_table)
        
        # Wedge analysis table
        wedge_table = wedge_analysis.to_latex(index=False, float_format="%.3f")
        with open(self.output_path.parent / "tables/wedge_analysis.tex", "w") as f:
            f.write(wedge_table)

def main():
    # Initialize classes
    pressure_analysis = PressureAnalysis()
    theory_calc = TheoryCalculations()
    viz = Visualization()
    
    # Process experimental data
    data_dir = Path("Lab 2/Data/Actual")
    angle_measurements = pd.read_csv("Lab 2/Scripts/angle_measurements.csv")
    
    experimental_results = []
    for file in data_dir.glob("*.txt"):
        results = pressure_analysis.process_file(file)
        experimental_results.append(results)
    
    exp_df = pd.DataFrame(experimental_results)
    
    # Create flow conditions table
    flow_conditions = pressure_analysis.create_flow_conditions_table(exp_df)
    
    # Analyze wedge flow
    wedge_analysis = theory_calc.analyze_wedge_flow(exp_df, angle_measurements)
    print("Wedge Analysis Results:")
    print(wedge_analysis)  # Add this line for debugging
    
    # Generate theoretical curves for Mach range
    mach_range = np.linspace(1.5, 3.5, 100)
    theoretical = theory_calc.theoretical_curves(mach_range)
    
    # Create plots and tables
    viz.plot_pressure_comparison(exp_df, theoretical)
    viz.plot_shock_angle_comparison(exp_df, theoretical, angle_measurements)
    viz.plot_reynolds_mach(exp_df)
    viz.create_latex_tables(flow_conditions, wedge_analysis)
    
    # Save results
    flow_conditions.to_csv('flow_conditions.csv', index=False)
    wedge_analysis.to_csv('wedge_analysis.csv', index=False)
    theoretical.to_csv('theoretical_results.csv', index=False)

if __name__ == "__main__":
    main()