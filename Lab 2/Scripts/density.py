import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

def plot_density_distributions(df, save_path=None):
    """
    Creates individual and combined plots of density distributions
    with proper labeling based on image filenames
    """
    # Set up the plot style
    plt.style.use('seaborn')
    
    # Create figure for combined plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color map for different images
    colors = plt.cm.tab20(np.linspace(0, 1, len(df.columns)//2))
    
    # Plot each profile with consistent colors for A and B lines
    for i, base_name in enumerate(set(col.split('_')[0] for col in df.columns)):
        # Get A and B profiles for this image
        a_col = f"{base_name}_A"
        b_col = f"{base_name}_B"
        
        if a_col in df.columns and b_col in df.columns:
            # Plot A profile (solid line)
            ax.plot(df[a_col], np.arange(len(df)), 
                   label=f'{base_name} - Line A',
                   color=colors[i], 
                   linestyle='-',
                   linewidth=2)
            
            # Plot B profile (dashed line)
            ax.plot(df[b_col], np.arange(len(df)), 
                   label=f'{base_name} - Line B',
                   color=colors[i], 
                   linestyle='--',
                   linewidth=2)
    
    # Customize the combined plot
    ax.set_title('Comparison of Density Distributions', fontsize=14, pad=20)
    ax.set_xlabel('Normalized Intensity', fontsize=12)
    ax.set_ylabel('Vertical Position (pixels)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def process_schlieren_images(folder_path):
    # Get all BMP files in the folder
    bmp_files = list(Path(folder_path).glob('*.bmp'))
    
    if not bmp_files:
        raise ValueError(f"No BMP files found in {folder_path}")
    
    # Dictionary to store all profiles
    all_profiles = {}
    
    for bmp_file in bmp_files:
        print(f"\nProcessing {bmp_file.name}")
        
        # Read the image
        img = cv2.imread(str(bmp_file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Display image for line selection
        fig = plt.figure(figsize=(12, 6))
        plt.imshow(gray, cmap='gray')
        plt.title(f'Select positions for lines A and B in {bmp_file.name}\nClick twice, then close window')
        
        # Get two x-positions from user clicks
        positions = plt.ginput(2)
        x_positions = [int(pos[0]) for pos in positions]
        plt.close(fig)
        
        # Get intensity profiles along vertical lines
        height = gray.shape[0]
        profile_a = gray[:, x_positions[0]] / 255.0
        profile_b = gray[:, x_positions[1]] / 255.0
        
        # Store profiles with meaningful names
        base_name = bmp_file.stem
        all_profiles[f"{base_name}_A"] = profile_a
        all_profiles[f"{base_name}_B"] = profile_b
        
        # Plot individual profiles for this image
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot A profile
        ax1.plot(profile_a, np.arange(height), 'b-', linewidth=2)
        ax1.set_title(f'{base_name}\nLine A Profile')
        ax1.set_xlabel('Normalized Intensity')
        ax1.set_ylabel('Vertical Position (pixels)')
        ax1.grid(True)
        
        # Plot B profile
        ax2.plot(profile_b, np.arange(height), 'r-', linewidth=2)
        ax2.set_title(f'{base_name}\nLine B Profile')
        ax2.set_xlabel('Normalized Intensity')
        ax2.set_ylabel('Vertical Position (pixels)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    # Create DataFrame
    df = pd.DataFrame(all_profiles)
    
    # Save to CSV
    csv_path = os.path.join(folder_path, 'density_profiles.csv')
    df.to_csv(csv_path, index=False)
    
    # Create and save combined plot
    plot_path = os.path.join(folder_path, 'density_distributions.png')
    plot_density_distributions(df, plot_path)
    
    return df

def main():
    # Get folder path from user
    folder_path = input("Enter the folder path containing BMP files: ")
    
    try:
        df = process_schlieren_images(folder_path)
        print(f"\nProcessing complete!")
        print(f"CSV file saved as: {os.path.join(folder_path, 'density_profiles.csv')}")
        print(f"Plot saved as: {os.path.join(folder_path, 'density_distributions.png')}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()