#!/usr/bin/env python
"""
Script to compare results from multiple PCR simulations.
Loads saved simulation states and compares their final mutant/WT ratios.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
from vcg_plot_saved_state import (
    ExtendedOligomerSystem, 
    melt_all_duplexes, 
    extract_cycle_data_ss
)

def find_simulation_file(timestamp, directory='plots'):
    """
    Find the simulation file with the given timestamp in the specified directory.
    
    Args:
        timestamp: Timestamp string to search for in filenames
        directory: Directory to search in (default: 'plots')
        
    Returns:
        Path to the found file, or None if not found
    """
    # Search for pkl files containing the timestamp
    search_pattern = os.path.join(directory, f"*{timestamp}*.pkl")
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        print(f"No files found matching timestamp {timestamp} in {directory}")
        return None
    
    # If multiple files match, use the first one
    if len(matching_files) > 1:
        print(f"Multiple files found for timestamp {timestamp}, using {matching_files[0]}")
    
    return matching_files[0]

def extract_final_ratio(state_file, label=None, flanking_offset=3):
    """
    Extract the final ratio of normalized mutant to WT concentration from a simulation.
    
    Args:
        state_file: Path to the pickle file with saved state
        label: Optional label for this simulation (default: extracted from filename)
        flanking_offset: Number of positions to offset flanking region inward from oligo bounds (default: 3)
        
    Returns:
        Dictionary with simulation data including final ratio and label
    """
    if not os.path.exists(state_file):
        print(f"Error: File {state_file} not found")
        return None
    
    # Extract a label from the filename if not provided
    if label is None:
        label = os.path.basename(state_file).split('.')[0]
    
    # Load the state
    print(f"Loading state from {state_file}...")
    state_data = ExtendedOligomerSystem.load_system_state(state_file)
    
    # Set flanking region based on the mutant-containing oligo
    flanking_region = None
    genome_length = state_data['genome_length']
    
    # Check if flanking_offset is valid
    if not flanking_offset > -1:
        print(f"Error: Flanking offset must be non-negative")
    else:
        # Get the initial state
        initial_state = None
        for state in state_data['cycle_states']:
            if state.get('label') == 'initial':
                initial_state = state
                break
        
        if initial_state:
            oligomers = initial_state.get('oligomers', [])
            concentrations = initial_state.get('concentrations', [])
            
            # Find the first mutant-containing oligo (assume there's only one)
            mutant_oligo = None
            
            for idx, oligo in enumerate(oligomers):
                if idx < len(concentrations) and concentrations[idx] > 0:
                    start, end, is_clockwise, is_duplex, is_mutant = oligo[:5]
                    
                    # Check if it's a mutant-containing oligo
                    if is_mutant:
                        mutant_oligo = oligo
                        break
            
            # Apply offset to the mutant oligo's start and end to create the flanking region
            if mutant_oligo:
                start, end = mutant_oligo[0], mutant_oligo[1]
                
                try:
                    # Handle wrap-around cases
                    if end >= start:  # Regular case
                        # Apply offset inward from both ends
                        flank_start = start + flanking_offset
                        flank_end = end - flanking_offset
                        
                        # Make sure we haven't inverted the region
                        if flank_end < flank_start:
                            oligo_length = end - start + 1
                            raise ValueError(
                                f"Flanking offset {flanking_offset} is too large for oligo of length {oligo_length}. "
                                f"This would create an invalid flanking region (start={flank_start}, end={flank_end}). "
                                f"Maximum valid offset would be {oligo_length // 2 - (0 if oligo_length % 2 == 0 else 1)}."
                            )
                    else:  # Wrapped case
                        # For wrapped oligos, apply offset in the direction toward the wrapped section
                        flank_start = (start + flanking_offset) % genome_length  # Move forward
                        flank_end = (end - flanking_offset) % genome_length      # Move backward
                        
                        # Check if we've inverted the wrapping
                        if flank_start < flank_end:
                            oligo_length = (genome_length - start) + end + 1
                            raise ValueError(
                                f"Flanking offset {flanking_offset} is too large for wrapped oligo of length {oligo_length}. "
                                f"This would create an invalid flanking region (start={flank_start}, end={flank_end}). "
                                f"Try a smaller offset value."
                            )
                    
                    flanking_region = (flank_start, flank_end)
                    print(f"Setting flanking region to {flanking_region} (offset {flanking_offset} from mutant oligo bounds {start},{end})")
                
                except ValueError as e:
                    print(f"Warning: {str(e)}")
                    flanking_region = None
    
    # Set the flanking region in the state data
    state_data['flanking_region'] = flanking_region
    
    # Now continue with the regular extraction procedure
    melted_data = melt_all_duplexes(state_data)
    
    # Extract cycle data
    cycle_data = extract_cycle_data_ss(state_data)
    melted_cycle_data = extract_cycle_data_ss(melted_data)
    
    cycles = cycle_data['cycles']
    data = cycle_data['data']
    melted_data = melted_cycle_data['data']
    
    if not cycles:
        print(f"No cycle data found in {state_file}")
        return None
    
    # Calculate total mutant and WT concentrations from melted data (last cycle)
    last_cycle_idx = len(cycles) - 1
    
    melted_mutant_cw = melted_data[last_cycle_idx]['mutant_cw_conc']
    melted_mutant_ccw = melted_data[last_cycle_idx]['mutant_ccw_conc']
    melted_wt_cw = melted_data[last_cycle_idx]['wt_cw_conc']
    melted_wt_ccw = melted_data[last_cycle_idx]['wt_ccw_conc']
    
    total_mutant_final = melted_mutant_cw + melted_mutant_ccw
    total_wt_final = melted_wt_cw + melted_wt_ccw
    
    # Calculate initial concentrations (cycle 0)
    melted_mutant_cw_init = melted_data[0]['mutant_cw_conc']
    melted_mutant_ccw_init = melted_data[0]['mutant_ccw_conc']
    melted_wt_cw_init = melted_data[0]['wt_cw_conc']
    melted_wt_ccw_init = melted_data[0]['wt_ccw_conc']
    
    total_mutant_init = melted_mutant_cw_init + melted_mutant_ccw_init
    total_wt_init = melted_wt_cw_init + melted_wt_ccw_init
    
    # Normalize to initial values
    normalized_mutant_final = total_mutant_final / total_mutant_init if total_mutant_init > 0 else 0
    normalized_wt_final = total_wt_final / total_wt_init if total_wt_init > 0 else 0
    
    # Calculate ratio of normalized values
    ratio = normalized_mutant_final / normalized_wt_final if normalized_wt_final > 0 else float('inf')
    
    # Get flanking region data if available
    if 'flanking_region' in state_data and state_data['flanking_region']:
        flanking_region = state_data['flanking_region']
        
        # Calculate mutant and WT with flanking regions
        mutant_cw_flanking = melted_data[last_cycle_idx]['mutant_cw_conc_flanking']
        mutant_ccw_flanking = melted_data[last_cycle_idx]['mutant_ccw_conc_flanking']
        wt_cw_flanking = melted_data[last_cycle_idx]['wt_cw_conc_flanking']
        wt_ccw_flanking = melted_data[last_cycle_idx]['wt_ccw_conc_flanking']
        
        total_mutant_flanking = mutant_cw_flanking + mutant_ccw_flanking
        total_wt_flanking = wt_cw_flanking + wt_ccw_flanking
        
        # Calculate initial concentrations with flanking regions
        mutant_cw_flanking_init = melted_data[0]['mutant_cw_conc_flanking']
        mutant_ccw_flanking_init = melted_data[0]['mutant_ccw_conc_flanking']
        wt_cw_flanking_init = melted_data[0]['wt_cw_conc_flanking']
        wt_ccw_flanking_init = melted_data[0]['wt_ccw_conc_flanking']
        
        total_mutant_flanking_init = mutant_cw_flanking_init + mutant_ccw_flanking_init
        total_wt_flanking_init = wt_cw_flanking_init + wt_ccw_flanking_init
        
        # Normalize flanking region values
        normalized_mutant_flanking = total_mutant_flanking / total_mutant_flanking_init if total_mutant_flanking_init > 0 else 0
        normalized_wt_flanking = total_wt_flanking / total_wt_flanking_init if total_wt_flanking_init > 0 else 0
        
        # Calculate ratio for flanking region
        ratio_flanking = normalized_mutant_flanking / normalized_wt_flanking if normalized_wt_flanking > 0 else float('inf')
    else:
        flanking_region = None
        ratio_flanking = None
        total_mutant_flanking = None
        total_wt_flanking = None
    
    # Get simulation parameters from filename if possible
    params = {}
    filename = os.path.basename(state_file)
    param_parts = filename.split('_')
    for part in param_parts:
        if '=' in part:
            key, value = part.split('=')
            try:
                # Convert numeric values
                params[key] = float(value)
            except ValueError:
                params[key] = value
    
    # Collect all data
    simulation_data = {
        'label': label,
        'filename': state_file,
        'num_cycles': len(cycles),
        'last_cycle': cycles[-1],
        'total_mutant_final': total_mutant_final,
        'total_wt_final': total_wt_final,
        'total_mutant_init': total_mutant_init,
        'total_wt_init': total_wt_init,
        'normalized_mutant_final': normalized_mutant_final,
        'normalized_wt_final': normalized_wt_final,
        'ratio': ratio,
        'flanking_region': flanking_region,
        'total_mutant_flanking': total_mutant_flanking,
        'total_wt_flanking': total_wt_flanking,
        'ratio_flanking': ratio_flanking,
        'params': params
    }
    
    return simulation_data

def plot_comparison(simulation_data_list, output_path=None, use_flanking=True, 
                    log_scale=False, sort_by_ratio=False, 
                    plot_both_ratios=True):
    """
    Plot a comparison of the final ratios from multiple simulations.
    
    Args:
        simulation_data_list: List of dictionaries with simulation data
        output_path: Path to save the figure
        use_flanking: Whether to use flanking region ratios (default: True)
        log_scale: Whether to use a log scale for y-axis (default: False)
        sort_by_ratio: Whether to sort bars by ratio (default: True)
        plot_both_ratios: Whether to plot both standard and flanking ratios (default: True)
    """
    if not simulation_data_list:
        print("No simulation data to plot")
        return
    
    # Create a figure with either one or two subplots
    fig, axs = plt.subplots(2 if plot_both_ratios else 1, 1, figsize=(12, 10 if plot_both_ratios else 6))
    
    # If there's only one subplot, make axs a list for consistent indexing
    if not plot_both_ratios:
        axs = [axs]
    
    # Extract data and prepare for plotting
    standard_ratios = []
    flanking_ratios = []
    labels = []
    
    for data in simulation_data_list:
        # Invert the ratios (WT/Mutant instead of Mutant/WT)
        if data['ratio'] > 0:
            standard_ratios.append(1.0 / data['ratio'])
        else:
            standard_ratios.append(float('inf'))
            
        if data['ratio_flanking'] is not None and data['ratio_flanking'] > 0:
            flanking_ratios.append(1.0 / data['ratio_flanking'])
        else:
            flanking_ratios.append(np.nan)
            
        labels.append(data['label'])
    
    # Sort data if requested
    if sort_by_ratio:
        # Create sorting pairs
        sorting_data = []
        for i in range(len(labels)):
            ratio_to_sort_by = standard_ratios[i]
            sorting_data.append((ratio_to_sort_by, standard_ratios[i], 
                                flanking_ratios[i], labels[i]))
        
        # Sort by the ratio
        sorting_data.sort()
        
        # Unpack sorted data
        standard_ratios = [item[1] for item in sorting_data]
        flanking_ratios = [item[2] for item in sorting_data]
        labels = [item[3] for item in sorting_data]
    
    # Plot standard ratios
    x = range(len(standard_ratios))
    axs[0].plot(x, standard_ratios, 'k-', marker='o', linewidth=2)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels, rotation=45, ha='right')
    
    if log_scale:
        axs[0].set_yscale('log')
    
    axs[0].set_ylabel('Inverse Final Ratio (Normalized WT / Normalized Mutant)')
    axs[0].set_title('Comparison of Final WT/Mutant Ratios - All Oligos')
    axs[0].grid(True, alpha=0.3)
    
    # Plot flanking ratios if we're showing both subplots
    if plot_both_ratios:
        axs[1].plot(x, flanking_ratios, 'k-', marker='s', linewidth=2)
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(labels, rotation=45, ha='right')
        
        if log_scale:
            axs[1].set_yscale('log')
        
        axs[1].set_ylabel('Final Ratio (Normalized WT / Normalized Mutant)')
        axs[1].set_title('Comparison of Final WT/Mutant Ratios - Flanking Region Only')
        axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
        
        # Save as PDF - replace extension with .pdf
        pdf_path = os.path.splitext(output_path)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Figure saved to {pdf_path}")
    else:
        plt.show()
    
    plt.close()


def compare_simulations(timestamps, output_dir='comparison_plots', directory='plots'):
    """
    Compare simulation results from multiple timestamps.
    
    Args:
        timestamps: List of timestamp strings to find simulation files
        output_dir: Directory to save output plots (default: 'comparison_plots')
        directory: Directory containing the simulation files (default: 'plots')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find and load simulation data
    simulation_data_list = []
    
    for timestamp in timestamps:
        state_file = find_simulation_file(timestamp, directory)
        if state_file:
            simulation_data = extract_final_ratio(state_file)
            if simulation_data:
                simulation_data_list.append(simulation_data)
    
    if not simulation_data_list:
        print("No valid simulation data found")
        return
    
    print(f"Found {len(simulation_data_list)} valid simulations")
    
    # Check if any simulations have flanking region data
    if any(data['ratio_flanking'] is not None for data in simulation_data_list):

        
        print("Generating combined ratio comparison plot...")
        plot_comparison(
            simulation_data_list,
            output_path=os.path.join(output_dir, "ratio_comparison_combined.png"),
            plot_both_ratios=True
        )

    
    print("Done!")

if __name__ == "__main__":
    # List of timestamps to compare
    # These should be replaced with actual timestamps of interest
    timestamps = [
        "0004_0918",  # Example timestamp 1
        "0003_0918",  # Example timestamp 2
        "0001_0918",  # Example timestamp 3
    ]
    
    compare_simulations(timestamps, output_dir='Fig4D_plots', directory='Fig4D_plots') 