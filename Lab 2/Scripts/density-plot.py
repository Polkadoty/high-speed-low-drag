import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

def plot_density_distributions(csv_path, save_path=None):
    """
    Creates individual and combined plots of density distributions
    with proper labeling based on image filenames
    """
    # Load the DataFrame from the CSV file
    df = pd.read_csv(csv_path)

    # Set up the plot style
    if 'seaborn' in plt.style.available:
        plt.style.use('seaborn')
    else:
        print("Warning: 'seaborn' style not found. Using default style.")
    
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

if __name__ == "__main__":
    csv_path = r"Lab 2\Scripts\density_profiles.csv"
    plot_density_distributions(csv_path)

