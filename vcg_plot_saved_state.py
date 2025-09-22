#!/usr/bin/env python
"""
Script to create advanced plots from saved PCR simulation states.
Generates various length distribution histograms, ridge plots, and heatmaps.
"""

from vcg_extended_fns import ExtendedOligomerSystem
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import argparse
import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42


def contains_flanking_region(oligo, flanking_region, genome_length):
    """
    Check if an oligo contains the full flanking region.
    
    Args:
        oligo: Tuple of (start, end, is_clockwise, is_duplex, is_mutant, ...)
        flanking_region: Tuple of (start, end) for the flanking region
        genome_length: Length of the circular genome
        
    Returns:
        Boolean indicating whether the oligo contains the full flanking region
    """
    start, end = oligo[:2]
    flank_start, flank_end = flanking_region
    
    # Handle cases where the flank wraps around the genome
    flank_wraps = flank_end < flank_start
    
    # Handle cases where the oligo wraps around the genome
    oligo_wraps = end < start
    
    # Case 1: Neither the oligo nor the flanking region wraps
    if not oligo_wraps and not flank_wraps:
        return start <= flank_start and end >= flank_end
    
    # Case 2: Flanking region wraps, oligo doesn't
    if not oligo_wraps and flank_wraps:
        # Oligo would need to span the entire genome to contain the flanking region
        return False
    
    # Case 3: Oligo wraps, flanking region doesn't
    if oligo_wraps and not flank_wraps:
        # When oligo wraps, we need to check if the flanking region falls entirely 
        # within the wrapped segment of the oligo
        
        # Split into two segments: start to end-of-genome and beginning-of-genome to end
        if flank_start >= start and flank_end <= genome_length - 1:
            # Flanking region is in first segment (start to end-of-genome)
            return True
        elif flank_start >= 0 and flank_end <= end:
            # Flanking region is in second segment (beginning-of-genome to end)
            return True
        elif flank_start >= start and flank_end <= end:
            # This could happen if the flanking region spans the wrap point
            return (flank_start >= start and flank_end <= end)
        else:
            return False
    
    # Case 4: Both oligo and flanking region wrap
    if oligo_wraps and flank_wraps:
        # For both wrapping, oligo must start at or before flank_start
        # and end at or after flank_end
        return start <= flank_start and end >= flank_end

def melt_all_duplexes(state_data):
    """
    Creates a new state where all duplexes are melted into single strands.
    
    Args:
        state_data: Loaded state data dictionary
        
    Returns:
        New state data with all duplexes melted into single strands
    """
    # Create a deep copy to avoid modifying the original
    import copy
    melted_state_data = copy.deepcopy(state_data)
    

    # Get genome length for melting threshold
    genome_length = state_data['genome_length']

    # Process each cycle state
    for i, state in enumerate(melted_state_data['cycle_states']):
        # Create temporary system with this state's data
        system = ExtendedOligomerSystem(genome_length)
        
        # Add all oligomers from this state to the system
        oligomers = state.get('oligomers', [])
        concentrations = state.get('concentrations', [])
        
        for j, oligo in enumerate(oligomers):
            if j < len(concentrations):
                # Unpack oligo tuple - assume all have 7 elements
                try:
                    start, end, is_clockwise, is_duplex, is_mutant, reactant1_idx, reactant2_idx = oligo
                    system.add_oligomer_type(start, end, is_clockwise, concentrations[j], 
                                           is_duplex, is_mutant, reactant1_idx, reactant2_idx)
                except ValueError as e:
                    print(f"Failed to unpack oligo at index {j}: {oligo}")
                    print(f"Oligo has {len(oligo)} elements, expected 7")
                    raise e
        
        # Melt all duplexes by setting min_stable_length above genome_length
        system.simulate_melt_cycle(min_stable_length=genome_length + 1, verbose=False)
        
        # Update the state with melted results
        new_oligomers = []
        new_concentrations = []
        
        # Get updated oligomers and concentrations
        for idx, oligo in enumerate(system.current_oligomers):
            new_oligomers.append(oligo)
            new_concentrations.append(system.concentrations[idx])
        
        # Update state
        melted_state_data['cycle_states'][i]['oligomers'] = new_oligomers
        melted_state_data['cycle_states'][i]['concentrations'] = new_concentrations
    
    return melted_state_data

def extract_cycle_data_ss(state_data):
    """
    Extract only single-stranded oligo data for each cycle from the saved state.
    
    Args:
        state_data: Loaded state data dictionary
        
    Returns:
        Dictionary with cycle data containing only single-stranded oligo information
    """
    results = {'cycles': [], 'data': []}
    
    # Check if we have cycle states
    if 'cycle_states' not in state_data or not state_data['cycle_states']:
        print("No cycle state data found")
        return results
        
    genome_length = state_data['genome_length']
    mutation_site = genome_length // 2
    cycle_states = state_data['cycle_states']
    
    # Get the flanking region from state_data if available
    flanking_region = state_data.get('flanking_region')
    if flanking_region:
        print(f"Using flanking region: {flanking_region}")
    
    # Helper function to calculate length
    def region_length(start, end):
        if end >= start:
            return end - start + 1
        else:
            return (genome_length - start) + end + 1
    
    # Helper function to calculate distance from 3' end to mutation site
    def distance_from_3prime(start, end, is_clockwise, mutation_site):
        # First check if mutation site is covered
        if is_clockwise:  # 5' at start, 3' at end
            if end >= start:  # Normal range
                if start <= mutation_site <= end:
                    return end - mutation_site  # Distance from 3' end to mutation
                return -1  # Mutation not covered
            else:  # Wraps around
                if start <= mutation_site or mutation_site <= end:
                    # Calculate distance accounting for wrap-around
                    if mutation_site >= start:
                        return (genome_length - 1 - mutation_site) + end
                    else:
                        return end - mutation_site
                return -1  # Mutation not covered
        else:  # CCW: 3' at start, 5' at end
            if end >= start:  # This should not happen for CCW
                if start <= mutation_site <= end:
                    return mutation_site - start  # Distance from 3' end to mutation
                return -1  # Mutation not covered
            else:  # Wraps around
                if start <= mutation_site or mutation_site <= end:
                    # Calculate distance accounting for wrap-around
                    if mutation_site >= start:
                        return mutation_site - start
                    else:
                        return mutation_site + (genome_length - start)
                return -1  # Mutation not covered
    
    # Process each cycle state
    for i, state in enumerate(cycle_states):
        label = state.get('label', f"State {i}")
        
        # Process label to get cycle number
        if label == "initial":
            cycle_num = 0
        elif label.startswith("cycle_"):
            try:
                cycle_num = int(label.split('_')[1])
            except (IndexError, ValueError):
                cycle_num = i
        else:
            cycle_num = i
            
        results['cycles'].append(cycle_num)
        
        # Process oligomers
        oligos = state.get('oligomers', [])
        concs = state.get('concentrations', [])
        
        # Data structures for this cycle
        cycle_data = {
            'ss_length_count': {},       # {length: [mutant_count, normal_count]}
            'ss_length_conc': {},        # {length: [mutant_conc, normal_conc]}
            'mutant_3prime_dist': [],    # List of distances for mutant oligos
            'wt_3prime_dist': [],        # List of distances for wildtype oligos
            'mutant_3prime_dist_w': [],  # Weighted by concentration
            'wt_3prime_dist_w': [],      # Weighted by concentration
            'mutant_3prime_dist_cw': [], # List of distances for CW mutant oligos
            'mutant_3prime_dist_ccw': [],# List of distances for CCW mutant oligos
            'wt_3prime_dist_cw': [],     # List of distances for CW wildtype oligos
            'wt_3prime_dist_ccw': [],    # List of distances for CCW wildtype oligos
            'mutant_3prime_dist_w_cw': [],  # Weighted by concentration for CW mutant oligos
            'mutant_3prime_dist_w_ccw': [], # Weighted by concentration for CCW mutant oligos
            'wt_3prime_dist_w_cw': [],      # Weighted by concentration for CW wildtype oligos
            'wt_3prime_dist_w_ccw': [],     # Weighted by concentration for CCW wildtype oligos
            'mutant_cw_conc': 0,         # Total concentration of CW mutant oligos
            'mutant_ccw_conc': 0,        # Total concentration of CCW mutant oligos
            'wt_cw_conc': 0,             # Total concentration of CW wildtype oligos
            'wt_ccw_conc': 0,            # Total concentration of CCW wildtype oligos
            'mutant_conc': 0,            # Total concentration of mutant oligos (CW + CCW)
            'wt_conc': 0,                # Total concentration of wildtype oligos (CW + CCW)
            'mutant_cw_conc_flanking': 0, # Concentration of CW mutant oligos with full flanking region
            'mutant_ccw_conc_flanking': 0, # Concentration of CCW mutant oligos with full flanking region
            'wt_cw_conc_flanking': 0,     # Concentration of CW wildtype oligos with full flanking region
            'wt_ccw_conc_flanking': 0,     # Concentration of CCW wildtype oligos with full flanking region
            'mutant_conc_flanking': 0,     # Total concentration of mutant oligos with full flanking region (CW + CCW)
            'wt_conc_flanking': 0          # Total concentration of wildtype oligos with full flanking region (CW + CCW)
        }
        
        # Process all single-stranded oligomers
        for idx, oligo in enumerate(oligos):
                
            conc = concs[idx]
            if conc <= 0:
                #print(f"Warning:Skipping oligo {oligo} with concentration {conc}")
                continue
                            
            start, end, is_clockwise, is_duplex, is_mutant = oligo[:5]
            
            # Skip duplexes
            if is_duplex:
                continue
            
            # Calculate length
            length = region_length(start, end)
            
            # Initialize counters if length not seen before
            for key in ['ss_length_count', 'ss_length_conc']:
                if length not in cycle_data[key]:
                    cycle_data[key][length] = [0, 0]  # [mutant, normal]
            
            # Update counters (ss and all are the same since we're only processing ss oligos)
            mutant_idx = 0 if is_mutant > 0 else 1
            cycle_data['ss_length_count'][length][mutant_idx] += 1
            cycle_data['ss_length_conc'][length][mutant_idx] += conc
            
            # Calculate distance from 3' end to mutation site
            dist = distance_from_3prime(start, end, is_clockwise, mutation_site)
            
            # If oligo covers mutation site, add to appropriate list
            if dist >= 0:
                # Check if oligo contains the full flanking region
                has_full_flanking = False
                if flanking_region:
                    has_full_flanking = contains_flanking_region(oligo, flanking_region, genome_length)

                if is_mutant:
                    cycle_data['mutant_3prime_dist'].append(dist)
                    # Add multiple entries based on concentration (weighted)
                    weight = int(conc * 100000)  # Scale for histogram
                    cycle_data['mutant_3prime_dist_w'].extend([dist] * weight)
                    
                    # Track CW/CCW orientation for mutant oligos
                    if is_clockwise:
                        cycle_data['mutant_cw_conc'] += conc
                        if has_full_flanking:
                            cycle_data['mutant_cw_conc_flanking'] += conc
                        # Track CW-specific distances
                        cycle_data['mutant_3prime_dist_cw'].append(dist)
                        cycle_data['mutant_3prime_dist_w_cw'].extend([dist] * weight)
                    else:
                        cycle_data['mutant_ccw_conc'] += conc
                        if has_full_flanking:
                            cycle_data['mutant_ccw_conc_flanking'] += conc
                        # Track CCW-specific distances
                        cycle_data['mutant_3prime_dist_ccw'].append(dist)
                        cycle_data['mutant_3prime_dist_w_ccw'].extend([dist] * weight)
                    
                    # Track total mutant concentration regardless of orientation
                    cycle_data['mutant_conc'] += conc
                    if has_full_flanking:
                        cycle_data['mutant_conc_flanking'] += conc
                else:
                    cycle_data['wt_3prime_dist'].append(dist)
                    # Add multiple entries based on concentration (weighted)
                    weight = int(conc * 100)  # Scale for histogram
                    cycle_data['wt_3prime_dist_w'].extend([dist] * weight)
                    
                    # Track CW/CCW orientation for wildtype oligos
                    if is_clockwise:
                        cycle_data['wt_cw_conc'] += conc
                        if has_full_flanking:
                            cycle_data['wt_cw_conc_flanking'] += conc
                        # Track CW-specific distances
                        cycle_data['wt_3prime_dist_cw'].append(dist)
                        cycle_data['wt_3prime_dist_w_cw'].extend([dist] * weight)
                    else:
                        cycle_data['wt_ccw_conc'] += conc
                        if has_full_flanking:
                            cycle_data['wt_ccw_conc_flanking'] += conc
                        # Track CCW-specific distances
                        cycle_data['wt_3prime_dist_ccw'].append(dist)
                        cycle_data['wt_3prime_dist_w_ccw'].extend([dist] * weight)
                    
                    # Track total wildtype concentration regardless of orientation
                    cycle_data['wt_conc'] += conc
                    if has_full_flanking:
                        cycle_data['wt_conc_flanking'] += conc
        
        results['data'].append(cycle_data)
        
    return results


def plot_length_histograms(cycle_data, include_duplexes=False, weighted=False, output_path=None):
    """
    Generate a comprehensive figure with length histograms, ridge plot, and heatmap.
    
    Args:
        cycle_data: Dictionary with cycle data
        include_duplexes: Whether to include oligos bound in duplexes; only affects plot title
        weighted: Whether to use concentration-weighted histograms
        output_path: Path to save the figure
    """
    if not cycle_data['cycles']:
        print("No cycle data to plot")
        return
    
    cycles = cycle_data['cycles']
    data = cycle_data['data']
    
    # Determine data source based on parameters
    data_key_base = 'ss_length'
    data_key = f'{data_key_base}_conc' if weighted else f'{data_key_base}_count'
    
    # Create figure with a revised layout (ridge plot + heatmap)
    fig = plt.figure(figsize=(15, 7))  # Wider figure to accommodate square subplots
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], figure=fig)
    
    # --- Ridge-style histogram (top) ---
    ridge_ax = fig.add_subplot(gs[0, 0])
    
    # Get all unique lengths across all cycles
    all_lengths = set()
    for cycle_idx in range(len(data)):
        lengths = sorted(data[cycle_idx][data_key].keys())
        all_lengths.update(lengths)
    
    # Sort lengths for consistent plotting
    all_lengths = sorted(all_lengths)
    if not all_lengths:
        print("No length data found")
        return
    
    # Create ridge-style histograms with normalization
    ridge_labels = []
    
    # Create palette for cycles
    sns.set_theme(style="white")
    palette = sns.color_palette("mako_r", len(cycles))
    
    # Find global max length for consistent binning
    max_length = max(all_lengths) if all_lengths else 0
    min_length = min(all_lengths) if all_lengths else 0
    
    # Prepare for plotting - similar to Figure C approach
    for i, cycle in enumerate(reversed(cycles)):
        cycle_idx = cycles.index(cycle)
        
        # Create arrays of mutant and normal values for all possible lengths
        mutant_values = []
        normal_values = []
        
        # Get histogram data
        for length in range(min_length, max_length + 1):
            if length in data[cycle_idx][data_key]:
                mutant_values.append(data[cycle_idx][data_key][length][0])  # Mutant count/conc
                normal_values.append(data[cycle_idx][data_key][length][1])  # Normal count/conc
            else:
                mutant_values.append(0)
                normal_values.append(0)
        
        # Create bin edges (like in Figure C)
        bin_edges = np.array(range(min_length, max_length + 2))  # +2 to include the max value
        
        # Calculate total values
        mutant_values = np.array(mutant_values)
        normal_values = np.array(normal_values)
        total_values = mutant_values + normal_values
        
        # Normalize to max height of 1.0 (like in Figure C)
        if np.max(total_values) > 0:
            scale_factor = 1.0 / np.max(total_values)
            mutant_values = mutant_values * scale_factor
            normal_values = normal_values * scale_factor
            total_values = total_values * scale_factor
        
        # Calculate vertical offset for this cycle
        offset = i * 1.2  # Ensure histograms don't touch
        
        # Create step plot for normal values - only plot non-zero values
        for j in range(len(normal_values)):
            if normal_values[j] > 0:  # Only plot if value is non-zero
                # Create the rectangle for this bin
                x_left = bin_edges[j]
                x_right = bin_edges[j + 1]
                y_bottom = offset
                y_top = normal_values[j] + offset
                
                # Plot as patch
                ridge_ax.fill_between(
                    [x_left, x_right], 
                    [y_bottom, y_bottom],
                    [y_top, y_top], 
                    color='#0072B2',  # Blue for normal
                    alpha=0.8,
                    linewidth=0,
                    step='mid'
                )
        
        # Create step plot for mutant values (stacked on top) - only plot non-zero values
        for j in range(len(mutant_values)):
            if mutant_values[j] > 0:  # Only plot if value is non-zero
                # Create the rectangle for this bin
                x_left = bin_edges[j]
                x_right = bin_edges[j + 1]
                y_bottom = normal_values[j] + offset
                y_top = total_values[j] + offset
                
                # Plot as patch
                ridge_ax.fill_between(
                    [x_left, x_right], 
                    [y_bottom, y_bottom],
                    [y_top, y_top], 
                    color='#E69F00',  # Orange for mutant
                    alpha=0.8,
                    linewidth=0,
                    step='mid'
                )
        
        # Add cycle label
        ridge_ax.text(
            max_length + 1.5,
            offset + 0.5,
            f"Cycle {cycle}",
            fontsize=9,
            va='center'
        )
        
        ridge_labels.append(f'Cycle {cycle}')
    
    # Set axis properties
    ridge_ax.set_xlim(min_length - 1, max_length + 5)
    ridge_ax.set_ylabel("Thermal Cycle Progression →")
    ridge_ax.set_xlabel("Oligomer Length")
    ridge_ax.set_title("Normalized Length Distribution Across Thermal Cycles")
    ridge_ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add a legend for mutant/normal
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#0072B2', alpha=0.8, label='Normal'),
        Patch(facecolor='#E69F00', alpha=0.8, label='Mutant')
    ]
    ridge_ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    # Hide y-ticks
    ridge_ax.set_yticks([])
    
    # --- Heatmap (bottom) ---
    heatmap_ax = fig.add_subplot(gs[0, 1])
    
    # Define bin size
    bin_size = 5
    
    # Create bin edges based on min and max lengths
    bin_edges = np.arange(min_length, max_length + bin_size, bin_size)
    if bin_edges[-1] < max_length:  # Make sure we include the max length
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin_size)
    
    # Create bin labels for y-axis (center of each bin) - switched from x-axis
    bin_centers = [(bin_edges[i] + bin_edges[i+1] - 1) // 2 for i in range(len(bin_edges) - 1)]
    bin_labels = [f"{bin_edges[i]}" for i in range(len(bin_edges) - 1)]
    
    # Prepare data for heatmap with binning
    # Need a matrix where columns are cycles and rows are length bins (switched from rows=cycles, cols=length bins)
    heatmap_data = np.zeros((len(bin_edges) - 1, len(cycles)))
    
    for i, cycle in enumerate(cycles):
        # Process each length and add to appropriate bin
        for length in range(min_length, max_length + 1):
            if length in data[i][data_key]:
                # Find which bin this length belongs to
                bin_idx = np.digitize(length, bin_edges) - 1
                if bin_idx >= 0 and bin_idx < len(bin_edges) - 1:
                    # Sum both mutant and normal values
                    heatmap_data[bin_idx, i] += data[i][data_key][length][0] + data[i][data_key][length][1]
    
    # Normalize within each column (cycle) - switched from row normalization
    for i in range(len(cycles)):
        col_max = np.max(heatmap_data[:, i])
        if col_max > 0:
            heatmap_data[:, i] = heatmap_data[:, i] / col_max
    
    # Create heatmap
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FFFFFF', '#808080', '#000000'])
    sns.heatmap(
        heatmap_data, 
        cmap=cmap,
        ax=heatmap_ax,
        cbar_kws={'label': 'Relative Concentration' if weighted else 'Relative Count'},
        xticklabels=[f"{c}" for c in cycles],  # Cycles on x-axis
        yticklabels=bin_labels  # Length bins on y-axis
    )
    
    # Set y-tick labels to be horizontal instead of vertical
    heatmap_ax.set_yticklabels(heatmap_ax.get_yticklabels(), rotation=0)
    
    # Flip y-axis so increasing length goes upward
    heatmap_ax.invert_yaxis()
    
    heatmap_ax.set_xlabel("Thermal Cycle")  # Switched from "Oligomer Length"
    heatmap_ax.set_ylabel("Oligomer Length")  # Switched from "Thermal Cycle"
    heatmap_ax.set_title("Heatmap of Length Distribution Over Cycles (Normalized per Cycle)")
    
    # Adjust layout
    plt.tight_layout()
    
    # Add figure title
    oligo_type = "All Oligomers" if include_duplexes else "Single-Stranded Oligomers Only"
    value_type = "Concentration-Weighted" if weighted else "Count-Based"
    fig.suptitle(f"Length Distribution of {oligo_type} ({value_type})", fontsize=14, y=0.98)
    plt.subplots_adjust(top=0.93)
    
    # Save or show
    if output_path:
        # Save as PNG
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
        
        # Save as PDF - replace extension with .pdf
        pdf_path = os.path.splitext(output_path)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Figure saved to {pdf_path}")
    else:
        plt.show()
    
    plt.close(fig)


def plot_mutation_distance(cycle_data, include_duplexes=False, log_color_scale=False, output_path=None):
    """
    Generate Figure C showing distribution of mutation distance from 3' end.
    Split into 6 panels for mutant CW, mutant CCW, wildtype CW, wildtype CCW,
    combined mutant (CW+CCW), and combined wildtype (CW+CCW) oligos.
    
    Args:
        cycle_data: Dictionary with cycle data
        include_duplexes: Whether to include oligos bound in duplexes; only affects plot title
        log_color_scale: Whether to use logarithmic scale for concentration-based coloring (default: False)
        output_path: Path to save the figure
    """

    oligo_type_for_plot_title = "All Oligomers" if include_duplexes else "Single-Stranded Oligomers Only"

    if not cycle_data['cycles']:
        print("No cycle data to plot")
        return
    
    cycles = cycle_data['cycles']
    data = cycle_data['data']
    
    # Create figure with a 2x3 layout (6 panels) - make wider for square subplots
    fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(22, 12), sharey=True)
    
    # Define the data sources and corresponding axes
    plot_configs = [
        {
            'data_field': 'mutant_3prime_dist_w_cw',
            'conc_field': 'mutant_cw_conc',  # Field containing the total concentration
            'axis': ax1,
            'title': "Mutant CW Oligos: Distance from 3' End to Mutation Site",
            'xlabel': "Distance from 3' End to Mutation",
            'color': '#fbb040'  # Uniform color for mutant
        },
        {
            'data_field': 'mutant_3prime_dist_w_ccw',
            'conc_field': 'mutant_ccw_conc',  # Field containing the total concentration
            'axis': ax2,
            'title': "Mutant CCW Oligos: Distance from 3' End to Mutation Site",
            'xlabel': "Distance from 3' End to Mutation",
            'color': '#fbb040'  # Uniform color for mutant
        },
        {
            'data_field': 'wt_3prime_dist_w_cw',
            'conc_field': 'wt_cw_conc',  # Field containing the total concentration
            'axis': ax3,
            'title': "Wildtype CW Oligos: Distance from 3' End to Wildtype Allele",
            'xlabel': "Distance from 3' End to Wildtype Allele",
            'color': '#00a651'  # Uniform color for wildtype
        },
        {
            'data_field': 'wt_3prime_dist_w_ccw',
            'conc_field': 'wt_ccw_conc',  # Field containing the total concentration
            'axis': ax4,
            'title': "Wildtype CCW Oligos: Distance from 3' End to Wildtype Allele",
            'xlabel': "Distance from 3' End to Wildtype Allele",
            'color': '#00a651'  # Uniform color for wildtype
        },
        {
            'data_field': 'mutant_3prime_dist_w',
            'conc_field': 'mutant_conc',  # Field containing the total concentration
            'axis': ax5,
            'title': "All Mutant Oligos: Distance from 3' End to Mutation Site",
            'xlabel': "Distance from 3' End to Mutation",
            'color': '#fbb040'  # Uniform color for mutant
        },
        {
            'data_field': 'wt_3prime_dist_w',
            'conc_field': 'wt_conc',  # Field containing the total concentration
            'axis': ax6,
            'title': "All Wildtype Oligos: Distance from 3' End to Wildtype Allele",
            'xlabel': "Distance from 3' End to Wildtype Allele",
            'color': '#00a651'  # Uniform color for wildtype
        }
    ]
    
    # Process each subplot
    for config in plot_configs:
        # --- Ridge plot for the current configuration ---
        # Prepare data for ridge plot
        ridge_data = []
        ridge_labels = []
        
        for i, cycle in enumerate(cycles):
            distances = data[i][config['data_field']]
            if not distances:
                continue
                
            # Create a frequency histogram for each cycle
            if len(distances) > 0:  # Make sure we have data
                hist, bin_edges = np.histogram(
                    distances, 
                    bins=range(max(distances) + 2)  # +2 to include the max value
                )
                
                # Normalize histogram so that max height is 1.0
                if np.sum(hist) > 0:
                    hist = hist / np.max(hist)
                
                # Store data for plotting - only for non-zero values
                for bin_idx in range(len(hist)):
                    if hist[bin_idx] > 0:  # Only store non-zero values
                        ridge_data.append({
                            'Distance': bin_edges[bin_idx],
                            'Frequency': hist[bin_idx],
                            'Cycle': cycle  # Store actual cycle number
                        })
            
            ridge_labels.append(cycle)  # Store actual cycle number
        
        # Create ridge plot
        if ridge_data:
            # Convert to a format seaborn can use
            df = {'Distance': [], 'Frequency': [], 'Cycle': []}
            for item in ridge_data:
                df['Distance'].append(item['Distance'])
                df['Frequency'].append(item['Frequency'])
                df['Cycle'].append(item['Cycle'])
            
            # Create ridge plot
            sns.set_theme(style="white")
            
            all_distances = sorted(set(df['Distance']))
            if not all_distances:
                print(f"No distance data found for {config['data_field']}")
                all_distances = [0]  # Default to avoid errors
            
            # Track the maximum vertical offset
            max_offset = 0
            cycle_to_data = {}
            
            # First, prepare the cycle data
            for cycle in cycles:
                # Filter data for this cycle
                cycle_distances = []
                cycle_freq = []
                
                for j in range(len(df['Distance'])):
                    if df['Cycle'][j] == cycle and df['Frequency'][j] > 0:  # Only include non-zero frequencies
                        cycle_distances.append(df['Distance'][j])
                        cycle_freq.append(df['Frequency'][j])
                
                if cycle_distances:
                    # Sort by distance
                    sorted_indices = np.argsort(cycle_distances)
                    cycle_distances = [cycle_distances[idx] for idx in sorted_indices]
                    cycle_freq = [cycle_freq[idx] for idx in sorted_indices]
                    
                    # We need to create separate segments for each non-zero region
                    # Group continuous non-zero regions
                    segments = []
                    current_segment = {'x': [], 'y': []}
                    
                    for j in range(len(cycle_distances)):
                        # Add this point
                        current_segment['x'].append(cycle_distances[j])
                        current_segment['y'].append(cycle_freq[j])
                        
                        # Check if this is the last point or if there's a gap to the next point
                        if j == len(cycle_distances) - 1 or cycle_distances[j+1] > cycle_distances[j] + 1:
                            # End of segment, save it if it has points
                            if current_segment['x']:
                                segments.append(current_segment)
                                current_segment = {'x': [], 'y': []}
                    
                    cycle_to_data[cycle] = {
                        'distances': cycle_distances,
                        'segments': segments
                    }
            
            # Now plot all cycles in proper order (cycle 0 at bottom, last cycle at top)
            for i, cycle in enumerate(cycles):
                # Calculate offset for this cycle (invert order so cycle 0 is at bottom)
                offset = i * 1.2  # Increase vertical spacing
                max_offset = max(max_offset, offset + 1.0)  # Track maximum offset
                
                # Add a faint guideline for every cycle, even if no data
                config['axis'].axhline(y=offset, color='gray', linestyle='-', alpha=0.15, linewidth=0.8)
                
                # Add cycle label for every cycle, even if no data
                config['axis'].text(
                    global_max_distance + 0.5 if 'global_max_distance' in locals() else max(all_distances) + 0.5, 
                    offset + 0.5,  # Position label at middle of histogram
                    f"Cycle {cycle}",
                    fontsize=9,
                    va='center'
                )
                
                # Use uniform color for all cycles
                bar_color = config['color']
                
                # Only plot data if this cycle has data
                if cycle in cycle_to_data:
                    # Plot each segment
                    for segment in cycle_to_data[cycle]['segments']:
                        if segment['x']:
                            # For each segment, we need left and right boundaries for each bin
                            x_values = []
                            y_values = []
                            
                            for j in range(len(segment['x'])):
                                # Create point at left edge of bin with a slight gap (5% of bin width)
                                x_values.append(segment['x'][j] - 0.45)  # Reduced from 0.5 to create gap
                                y_values.append(segment['y'][j])
                                
                                # Create point at right edge of bin with a slight gap
                                x_values.append(segment['x'][j] + 0.45)  # Reduced from 0.5 to create gap
                                y_values.append(segment['y'][j])
                            
                            # Plot this segment using uniform color
                            config['axis'].fill_between(
                                x_values,
                                offset,
                                [v + offset for v in y_values],
                                alpha=0.8,
                                color=bar_color,
                                step='mid',
                                linewidth=0  # Remove the default edge
                            )
                            
                            # Add black outline to each individual bar
                            for j in range(len(segment['x'])):
                                # Draw rectangle outline for each bar
                                rect = plt.Rectangle(
                                    (segment['x'][j] - 0.45, offset),  # Left, bottom
                                    0.9,  # Width (reduced from 1.0 to create gap)
                                    segment['y'][j],  # Height
                                    fill=False,
                                    edgecolor='black',
                                    linewidth=0.5,
                                    zorder=3
                                )
                                config['axis'].add_patch(rect)
            
            # Store the maximum distance for this subplot 
            # (we'll use it later to set consistent x-axis limits)
            config['max_distance'] = max(all_distances) if all_distances else 0
            # Store the maximum y offset for this subplot
            config['max_y_offset'] = max_offset
            
            config['axis'].set_xlabel(config['xlabel'])
            config['axis'].set_title(config['title'] + " (" + oligo_type_for_plot_title + ")")
            config['axis'].grid(axis='x', linestyle='--', alpha=0.3)
            # Hide y-ticks
            config['axis'].set_yticks([])
            
        else:
            # No data for this subplot, still need to setup a proper empty plot
            # Create guidelines and labels for all cycles
            max_offset = 0
            for i, cycle in enumerate(cycles):
                offset = i * 1.2  # Increase vertical spacing
                max_offset = max(max_offset, offset + 1.0)
                
                # Add a faint guideline for every cycle
                config['axis'].axhline(y=offset, color='gray', linestyle='-', alpha=0.15, linewidth=0.8)
                
                # Add cycle label for every cycle
                config['axis'].text(
                    3,  # Default position when no data
                    offset + 0.5,
                    f"Cycle {cycle}",
                    fontsize=9,
                    va='center'
                )
            
            config['max_distance'] = 0
            config['max_y_offset'] = max_offset
            config['axis'].set_xlabel(config['xlabel'])
            config['axis'].set_title(config['title'] + " (" + oligo_type_for_plot_title + ")")
            config['axis'].grid(axis='x', linestyle='--', alpha=0.3)
            config['axis'].set_yticks([])
    
    # Find the global maximum distance across all subplots
    global_max_distance = max(config.get('max_distance', 0) for config in plot_configs)
    
    # Find the global maximum y offset across all subplots
    global_max_y_offset = max(config.get('max_y_offset', 0) for config in plot_configs)
    
    # Set identical x-axis and y-axis limits for all subplots
    for config in plot_configs:
        config['axis'].set_xlim(-0.5, global_max_distance + 3)
        config['axis'].set_ylim(0, global_max_y_offset + 0.5)
    
    # Add ylabel only to the left axes
    ax1.set_ylabel("Thermal Cycle Progression →")
    ax3.set_ylabel("Thermal Cycle Progression →")
    
    # Add ylabel to the rightmost panels as well
    ax5.set_ylabel("Thermal Cycle Progression →")
    ax6.set_ylabel("Thermal Cycle Progression →")
    
    # Add figure title
    fig.suptitle("Distribution of Distance from 3' End to Mutation/Wildtype Site by Orientation (" + oligo_type_for_plot_title + ")", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save or show
    if output_path:
        # Save as PNG
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
        
        # Save as PDF - replace extension with .pdf
        pdf_path = os.path.splitext(output_path)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Figure saved to {pdf_path}")
    else:
        plt.show()
    
    plt.close(fig)

def plot_cycle_summary(state_data, cycle_data, melted_cycle_data, normalized_max=2.5, output_path=None):
    """
    Generate Figure Z with four panels:
    1. Oligomer counts by type vs cycle number
    2a. Concentration of mutant and WT alleles by orientation (single-stranded only)
    2b. Concentration of mutant and WT alleles by orientation (from melted data)
    3. Normalized total mutant and WT concentrations
    
    Args:
        state_data: Raw state data dictionary
        cycle_data: Dictionary with cycle data
        melted_cycle_data: Dictionary with cycle data from melted duplexes
        normalized_max: Maximum normalized concentration for y-axis scaling
        output_path: Path to save the figure
    """
    if not cycle_data['cycles']:
        print("No cycle data to plot")
        return
    
    cycles = cycle_data['cycles']
    data = cycle_data['data']
    melted_data = melted_cycle_data['data']
    
    # Create figure with a modified grid layout - now 2x3
    fig = plt.figure(figsize=(15, 10))  # Wider figure to accommodate extra panel
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1])
    
    # --- Panel 1: Oligomer counts by type vs cycle number ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    ss_normal = []
    ss_mutant = []
    duplex_normal = []
    duplex_one_mut = []
    duplex_both_mut = []
    total_conc = []
    
    state_cycles = []
    for state in state_data['cycle_states']:
        label = state.get('label', '')
        
        # Process label to get cycle number
        if label == "initial":
            cycle_num = 0
        elif label.startswith("cycle_"):
            cycle_num = int(label.split('_')[1])
        
        state_cycles.append(cycle_num)
        
        # Initialize counters for this cycle
        cycle_ss_normal = 0
        cycle_ss_mutant = 0
        cycle_duplex_normal = 0
        cycle_duplex_one_mut = 0
        cycle_duplex_both_mut = 0
        cycle_total = 0
        
        # Process all oligomers in this cycle
        oligomers = state.get('oligomers', [])
        concentrations = state.get('concentrations', [])
        
        for idx, oligo in enumerate(oligomers):                
            conc = concentrations[idx]
            if conc <= 0:
                continue
                
            cycle_total += conc
            
            # Parse oligomer tuple
            start, end, is_clockwise, is_duplex, is_mutant = oligo[:5]
            
            if not is_duplex:
                # Single-stranded oligomers
                if is_mutant:
                    cycle_ss_mutant += conc
                else:
                    cycle_ss_normal += conc
            else:
                # Duplex oligomers
                # Categorize the duplex based on mutations
                if is_mutant == 3:
                    cycle_duplex_both_mut += conc
                elif is_mutant == 1 or is_mutant == 2:
                    cycle_duplex_one_mut += conc
                else:
                    cycle_duplex_normal += conc
        
        # Add data for this cycle
        ss_normal.append(cycle_ss_normal)
        ss_mutant.append(cycle_ss_mutant)
        duplex_normal.append(cycle_duplex_normal)
        duplex_one_mut.append(cycle_duplex_one_mut)
        duplex_both_mut.append(cycle_duplex_both_mut)
        total_conc.append(cycle_total)    
    
    # Plot oligomer counts
    ax1.plot(state_cycles, np.array(ss_normal) + np.array(ss_mutant), 's--', color='black', label='SS Normal')
    ax1.plot(state_cycles, np.array(duplex_normal) + np.array(duplex_one_mut) + np.array(duplex_both_mut), 'o-', color='black', label='Duplex Normal-Normal')
    ax1.plot(state_cycles, np.array(total_conc), 'o-', color='grey', label='Total Concentration')
    
    ax1.set_ylabel('Concentration')
    ax1.set_xlabel('Thermal Cycle')
    ax1.set_title('Oligomer Concentrations by Type')
    ax1.legend(loc='upper center', fontsize=8, bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax1.grid(True, alpha=0.3)
    
    # --- Panel 2a: Original SS Allele concentration (single-stranded only) ---
    ax2a = fig.add_subplot(gs[0, 1])
    
    mutant_cw = np.zeros(len(cycles))
    mutant_ccw = np.zeros(len(cycles))
    wt_cw = np.zeros(len(cycles))
    wt_ccw = np.zeros(len(cycles))
    
    for i, cycle in enumerate(cycles):
        # Use the tracked orientation data (single-stranded only)
        mutant_cw[i] = data[i]['mutant_cw_conc']
        mutant_ccw[i] = data[i]['mutant_ccw_conc']
        wt_cw[i] = data[i]['wt_cw_conc']
        wt_ccw[i] = data[i]['wt_ccw_conc']
    
    # Plot mutant and WT allele concentration with different symbols for CW and CCW
    ax2a.plot(cycles, mutant_cw + mutant_ccw, 'o-', color='#fbb040', label='Total Mutant')
    ax2a.plot(cycles, wt_cw + wt_ccw, 'o-', color='#00a651', label='Total WT')
    ax2a.plot(cycles, mutant_cw, '^--', color='#fbb040', alpha=0.6, label='Mutant CW')  # Triangle for CW
    ax2a.plot(cycles, mutant_ccw, 's:', color='#fbb040', alpha=0.6, label='Mutant CCW')  # Square for CCW
    ax2a.plot(cycles, wt_cw, '^--', color='#00a651', alpha=0.6, label='WT CW')  # Triangle for CW
    ax2a.plot(cycles, wt_ccw, 's:', color='#00a651', alpha=0.6, label='WT CCW')  # Square for CCW
    
    ax2a.set_yscale('log')  # Set y-axis to log scale
    
    # Ensure y-axis has tick label at the top and extends downward
    if len(cycles) > 0:
        # Find min non-zero value to extend downward
        all_values = np.concatenate([
            mutant_cw[mutant_cw > 0], 
            mutant_ccw[mutant_ccw > 0], 
            wt_cw[wt_cw > 0], 
            wt_ccw[wt_ccw > 0]
        ])
        
        if len(all_values) > 0:
            min_val = np.min(all_values) * 0.5  # Extend 50% lower
            max_val = np.max([
                np.max(mutant_cw + mutant_ccw) if len(mutant_cw + mutant_ccw) > 0 else 0,
                np.max(wt_cw + wt_ccw) if len(wt_cw + wt_ccw) > 0 else 0
            ]) * 1.3  # Extend 20% higher
            
            ax2a.set_ylim(min_val, max_val)
            
            # Force ticks at min and max values
            ax2a.yaxis.set_major_locator(plt.LogLocator(numticks=10))
            ax2a.yaxis.set_minor_locator(plt.LogLocator(subs='all', numticks=20))
    
    ax2a.set_ylabel('Concentration (log scale)')
    ax2a.set_xlabel('Thermal Cycle')
    ax2a.set_title('SS Oligos: Allele Concentration by Orientation')
    ax2a.legend(loc='upper center', fontsize=8, bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax2a.grid(True, alpha=0.3)
    
    # --- Panel 2b: Melted Allele concentration from melted_cycle_data ---
    ax2b = fig.add_subplot(gs[0, 2])
    
    # Same as Panel 2a but using melted_data
    melted_mutant_cw = np.zeros(len(cycles))
    melted_mutant_ccw = np.zeros(len(cycles))
    melted_wt_cw = np.zeros(len(cycles))
    melted_wt_ccw = np.zeros(len(cycles))
    
    for i, cycle in enumerate(cycles):
        # Use the tracked orientation data from melted cycle data
        melted_mutant_cw[i] = melted_data[i]['mutant_cw_conc']
        melted_mutant_ccw[i] = melted_data[i]['mutant_ccw_conc']
        melted_wt_cw[i] = melted_data[i]['wt_cw_conc']
        melted_wt_ccw[i] = melted_data[i]['wt_ccw_conc']
    
    # Plot mutant and WT allele concentration with different symbols for CW and CCW
    ax2b.plot(cycles, melted_mutant_cw + melted_mutant_ccw, 'o-', color='#fbb040', label='Total Mutant')
    ax2b.plot(cycles, melted_wt_cw + melted_wt_ccw, 'o-', color='#00a651', label='Total WT')
    ax2b.plot(cycles, melted_mutant_cw, '^--', color='#fbb040', alpha=0.6, label='Mutant CW')  # Triangle for CW
    ax2b.plot(cycles, melted_mutant_ccw, 's:', color='#fbb040', alpha=0.6, label='Mutant CCW')  # Square for CCW
    ax2b.plot(cycles, melted_wt_cw, '^--', color='#00a651', alpha=0.6, label='WT CW')  # Triangle for CW
    ax2b.plot(cycles, melted_wt_ccw, 's:', color='#00a651', alpha=0.6, label='WT CCW')  # Square for CCW
    
    ax2b.set_yscale('log')  # Set y-axis to log scale
    
    # Ensure y-axis has tick label at the top and extends downward
    if len(cycles) > 0:
        # Find min non-zero value to extend downward
        all_values = np.concatenate([
            melted_mutant_cw[melted_mutant_cw > 0], 
            melted_mutant_ccw[melted_mutant_ccw > 0], 
            melted_wt_cw[melted_wt_cw > 0], 
            melted_wt_ccw[melted_wt_ccw > 0]
        ])
        
        if len(all_values) > 0:
            min_val = np.min(all_values) * 0.5  # Extend 50% lower
            max_val = np.max([
                np.max(melted_mutant_cw + melted_mutant_ccw) if len(melted_mutant_cw + melted_mutant_ccw) > 0 else 0,
                np.max(melted_wt_cw + melted_wt_ccw) if len(melted_wt_cw + melted_wt_ccw) > 0 else 0
            ]) * 1.3  # Extend 20% higher
            
            ax2b.set_ylim(min_val, max_val)
            
            # Force ticks at min and max values
            ax2b.yaxis.set_major_locator(plt.LogLocator(numticks=10))
            ax2b.yaxis.set_minor_locator(plt.LogLocator(subs='all', numticks=20))
    
    ax2b.set_ylabel('Concentration (log scale)')
    ax2b.set_xlabel('Thermal Cycle')
    ax2b.set_title('SS+Melted Duplexes: Allele Concentration by Orientation')
    ax2b.legend(loc='upper center', fontsize=8, bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax2b.grid(True, alpha=0.3)
    
    # --- Panel 3: Normalized total mutant and WT concentrations ---
    ax3 = fig.add_subplot(gs[1, 0])  # Span all three columns
    
    # Calculate total mutant and WT concentrations from melted data
    total_mutant = np.zeros(len(cycles))
    total_wt = np.zeros(len(cycles))
    
    for i in range(len(cycles)):
        total_mutant[i] = melted_mutant_cw[i] + melted_mutant_ccw[i]
        total_wt[i] = melted_wt_cw[i] + melted_wt_ccw[i]
    
    # Normalize to initial values
    if len(cycles) > 0 and total_mutant[0] > 0 and total_wt[0] > 0:
        normalized_mutant = total_mutant / total_mutant[0]
        normalized_wt = total_wt / total_wt[0]
        
        # Plot normalized concentrations
        ax3.plot(cycles, normalized_mutant, 'o-', color='#fbb040', label='Normalized Total Mutant')
        ax3.plot(cycles, normalized_wt, 'o-', color='#00a651', label='Normalized Total WT')
        ax3.set_ylim(0.8, normalized_max)
    
    ax3.set_xlabel('Thermal Cycle')
    ax3.set_ylabel('Normalized Concentration')
    ax3.set_title('Normalized Total Mutant and WT Concentrations')
    ax3.legend(loc='upper center', fontsize=8, bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax3.grid(True, alpha=0.3)
    
    # Add figure title
    fig.suptitle("Figure Z: Thermal Cycle Summary", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save or show
    if output_path:
        # Save as PNG
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
        
        # Save as PDF - replace extension with .pdf
        pdf_path = os.path.splitext(output_path)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Figure saved to {pdf_path}")
    else:
        plt.show()
    
    plt.close(fig)

def plot_cycle_summary_flanking(state_data, cycle_data, melted_cycle_data, normalized_max=8, output_path=None):
    """
    Generate Figure Z with four panels, showing flanking region concentrations:
    1. Oligomer counts by type vs cycle number
    2a. Concentration of mutant and WT alleles with flanking regions by orientation (single-stranded only)
    2b. Concentration of mutant and WT alleles with flanking regions by orientation (from melted data)
    3. Normalized total mutant and WT concentrations with flanking regions
    
    Args:
        state_data: Raw state data dictionary
        cycle_data: Dictionary with cycle data
        melted_cycle_data: Dictionary with cycle data from melted duplexes
        normalized_max: Maximum normalized concentration for y-axis scaling
        output_path: Path to save the figure
    """
    if not cycle_data['cycles']:
        print("No cycle data to plot")
        return
    
    cycles = cycle_data['cycles']
    data = cycle_data['data']
    melted_data = melted_cycle_data['data']
    
    # Create figure with a modified grid layout - now 2x3
    fig = plt.figure(figsize=(15, 10))  # Wider figure to accommodate extra panel
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1])
    
    # --- Panel 1: Oligomer counts by type vs cycle number ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    ss_normal = []
    ss_mutant = []
    duplex_normal = []
    duplex_one_mut = []
    duplex_both_mut = []
    total_conc = []
    
    # Get flanking region from state_data
    flanking_region = state_data.get('flanking_region')
    genome_length = state_data['genome_length']
    
    state_cycles = []
    for state in state_data['cycle_states']:
        label = state.get('label', '')
        
        # Process label to get cycle number
        if label == "initial":
            cycle_num = 0
        elif label.startswith("cycle_"):
            cycle_num = int(label.split('_')[1])
        
        state_cycles.append(cycle_num)
        
        # Initialize counters for this cycle
        cycle_ss_normal = 0
        cycle_ss_mutant = 0
        cycle_duplex_normal = 0
        cycle_duplex_one_mut = 0
        cycle_duplex_both_mut = 0
        cycle_total = 0
        
        # Process all oligomers in this cycle
        oligomers = state.get('oligomers', [])
        concentrations = state.get('concentrations', [])
        
        for idx, oligo in enumerate(oligomers):                
            conc = concentrations[idx]
            if conc <= 0:
                continue
                
            # Parse oligomer tuple
            start, end, is_clockwise, is_duplex, is_mutant = oligo[:5]
            
            # Check if oligo contains the full flanking region
            has_full_flanking = False
            if flanking_region:
                has_full_flanking = contains_flanking_region(oligo, flanking_region, genome_length)
            
            # Only include oligos with flanking region in total concentration
            if has_full_flanking:
                cycle_total += conc
            
            if not is_duplex:
                # Single-stranded oligomers
                if has_full_flanking:
                    if is_mutant:
                        cycle_ss_mutant += conc
                    else:
                        cycle_ss_normal += conc
            else:
                # Duplex oligomers
                # Categorize the duplex based on mutations
                if has_full_flanking:
                    if is_mutant == 3:
                        cycle_duplex_both_mut += conc
                    elif is_mutant == 1 or is_mutant == 2:
                        cycle_duplex_one_mut += conc
                    else:
                        cycle_duplex_normal += conc
        
        # Add data for this cycle
        ss_normal.append(cycle_ss_normal)
        ss_mutant.append(cycle_ss_mutant)
        duplex_normal.append(cycle_duplex_normal)
        duplex_one_mut.append(cycle_duplex_one_mut)
        duplex_both_mut.append(cycle_duplex_both_mut)
        total_conc.append(cycle_total)    
    
    # Plot oligomer counts
    ax1.plot(state_cycles, np.array(ss_normal) + np.array(ss_mutant), 's--', color='grey', label='SS Normal')
    ax1.plot(state_cycles, np.array(duplex_normal) + np.array(duplex_one_mut) + np.array(duplex_both_mut), 'o-', color='grey', label='Duplex Normal-Normal')
    ax1.plot(state_cycles, np.array(total_conc), 'o-', color='black', label='Total Concentration')

    
    ax1.set_ylabel('Concentration')
    ax1.set_xlabel('Thermal Cycle')
    ax1.set_title('Oligomer Concentrations by Type (with Flanking Region)')
    ax1.legend(loc='upper center', fontsize=8, bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax1.grid(True, alpha=0.3)
    
    # --- Panel 2a: Original SS Allele concentration with flanking regions (single-stranded only) ---
    ax2a = fig.add_subplot(gs[0, 1])
    
    mutant_cw = np.zeros(len(cycles))
    mutant_ccw = np.zeros(len(cycles))
    wt_cw = np.zeros(len(cycles))
    wt_ccw = np.zeros(len(cycles))
    
    for i, cycle in enumerate(cycles):
        # Use the tracked orientation data with flanking regions (single-stranded only)
        mutant_cw[i] = data[i]['mutant_cw_conc_flanking']
        mutant_ccw[i] = data[i]['mutant_ccw_conc_flanking']
        wt_cw[i] = data[i]['wt_cw_conc_flanking']
        wt_ccw[i] = data[i]['wt_ccw_conc_flanking']
    
    # Plot mutant and WT allele concentration with different symbols for CW and CCW
    ax2a.plot(cycles, mutant_cw + mutant_ccw, 'o-', color='#fbb040', label='Total Mutant')
    ax2a.plot(cycles, wt_cw + wt_ccw, 'o-', color='#00a651', label='Total WT')
    ax2a.plot(cycles, mutant_cw, '^--', color='#fbb040', alpha=0.6, label='Mutant CW')  # Triangle for CW
    ax2a.plot(cycles, mutant_ccw, 's:', color='#fbb040', alpha=0.6, label='Mutant CCW')  # Square for CCW
    ax2a.plot(cycles, wt_cw, '^--', color='#00a651', alpha=0.6, label='WT CW')  # Triangle for CW
    ax2a.plot(cycles, wt_ccw, 's:', color='#00a651', alpha=0.6, label='WT CCW')  # Square for CCW
    
    ax2a.set_yscale('log')  # Set y-axis to log scale
    
    # Ensure y-axis has tick label at the top and extends downward
    if len(cycles) > 0:
        # Find min non-zero value to extend downward
        all_values = np.concatenate([
            mutant_cw[mutant_cw > 0], 
            mutant_ccw[mutant_ccw > 0], 
            wt_cw[wt_cw > 0], 
            wt_ccw[wt_ccw > 0]
        ])
        
        if len(all_values) > 0:
            min_val = np.min(all_values) * 0.5  # Extend 50% lower
            max_val = np.max([
                np.max(mutant_cw + mutant_ccw) if len(mutant_cw + mutant_ccw) > 0 else 0,
                np.max(wt_cw + wt_ccw) if len(wt_cw + wt_ccw) > 0 else 0
            ]) * 1.3  # Extend 20% higher
            
            ax2a.set_ylim(min_val, max_val)
            
            # Force ticks at min and max values
            ax2a.yaxis.set_major_locator(plt.LogLocator(numticks=10))
            ax2a.yaxis.set_minor_locator(plt.LogLocator(subs='all', numticks=20))
    
    ax2a.set_ylabel('Concentration (log scale)')
    ax2a.set_xlabel('Thermal Cycle')
    ax2a.set_title('SS Oligos: Allele with Flanking Region Concentration')
    ax2a.legend(loc='upper center', fontsize=8, bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax2a.grid(True, alpha=0.3)
    
    # --- Panel 2b: Melted Allele concentration with flanking regions from melted_cycle_data ---
    ax2b = fig.add_subplot(gs[0, 2])
    
    # Same as Panel 2a but using melted_data
    melted_mutant_cw = np.zeros(len(cycles))
    melted_mutant_ccw = np.zeros(len(cycles))
    melted_wt_cw = np.zeros(len(cycles))
    melted_wt_ccw = np.zeros(len(cycles))
    
    for i, cycle in enumerate(cycles):
        # Use the tracked orientation data with flanking regions from melted cycle data
        melted_mutant_cw[i] = melted_data[i]['mutant_cw_conc_flanking']
        melted_mutant_ccw[i] = melted_data[i]['mutant_ccw_conc_flanking']
        melted_wt_cw[i] = melted_data[i]['wt_cw_conc_flanking']
        melted_wt_ccw[i] = melted_data[i]['wt_ccw_conc_flanking']
    
    # Plot mutant and WT allele concentration with different symbols for CW and CCW
    ax2b.plot(cycles, melted_mutant_cw + melted_mutant_ccw, 'o-', color='#fbb040', label='Total Mutant')
    ax2b.plot(cycles, melted_wt_cw + melted_wt_ccw, 'o-', color='#00a651', label='Total WT')
    ax2b.plot(cycles, melted_mutant_cw, '^--', color='#fbb040', alpha=0.6, label='Mutant CW')  # Triangle for CW
    ax2b.plot(cycles, melted_mutant_ccw, 's:', color='#fbb040', alpha=0.6, label='Mutant CCW')  # Square for CCW
    ax2b.plot(cycles, melted_wt_cw, '^--', color='#00a651', alpha=0.6, label='WT CW')  # Triangle for CW
    ax2b.plot(cycles, melted_wt_ccw, 's:', color='#00a651', alpha=0.6, label='WT CCW')  # Square for CCW
    
    ax2b.set_yscale('log')  # Set y-axis to log scale
    
    # Ensure y-axis has tick label at the top and extends downward
    if len(cycles) > 0:
        # Find min non-zero value to extend downward
        all_values = np.concatenate([
            melted_mutant_cw[melted_mutant_cw > 0], 
            melted_mutant_ccw[melted_mutant_ccw > 0], 
            melted_wt_cw[melted_wt_cw > 0], 
            melted_wt_ccw[melted_wt_ccw > 0]
        ])
        
        if len(all_values) > 0:
            min_val = np.min(all_values) * 0.5  # Extend 50% lower
            max_val = np.max([
                np.max(melted_mutant_cw + melted_mutant_ccw) if len(melted_mutant_cw + melted_mutant_ccw) > 0 else 0,
                np.max(melted_wt_cw + melted_wt_ccw) if len(melted_wt_cw + melted_wt_ccw) > 0 else 0
            ]) * 1.3  # Extend 20% higher
            
            ax2b.set_ylim(min_val, max_val)
            
            # Force ticks at min and max values
            ax2b.yaxis.set_major_locator(plt.LogLocator(numticks=10))
            ax2b.yaxis.set_minor_locator(plt.LogLocator(subs='all', numticks=20))
    
    ax2b.set_ylabel('Concentration (log scale)')
    ax2b.set_xlabel('Thermal Cycle')
    ax2b.set_title('SS+Melted Duplexes: Allele with Flanking Region')
    ax2b.legend(loc='upper center', fontsize=8, bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax2b.grid(True, alpha=0.3)
    
    # --- Panel 5: Total mutant and WT concentrations normalized by their initial amounts (including duplexes) in flanking region ---
    ax5 = fig.add_subplot(gs[1, 0:1])  # Span all three columns for clarity
    
    # Use the melted data to calculate total mutant and WT concentrations
    total_mutant = np.zeros(len(cycles))
    total_wt = np.zeros(len(cycles))
    
    for i in range(len(cycles)):
        total_mutant[i] = melted_mutant_cw[i] + melted_mutant_ccw[i]
        total_wt[i] = melted_wt_cw[i] + melted_wt_ccw[i]
    
    # Normalize by initial amounts
    if total_mutant[0] > 0:
        total_mutant_normalized = total_mutant / total_mutant[0]
    else:
        total_mutant_normalized = np.zeros(len(cycles))
    
    if total_wt[0] > 0:
        total_wt_normalized = total_wt / total_wt[0]
    else:
        total_wt_normalized = np.zeros(len(cycles))
    
    # Plot normalized concentrations
    ax5.plot(cycles, total_mutant_normalized, 'o-', color='#fbb040', label='Mutant (Normalized)')
    ax5.plot(cycles, total_wt_normalized, 'o-', color='#00a651', label='WT (Normalized)')
    ax5.set_ylim(0.5, normalized_max)
    
    ax5.set_xlabel('Thermal Cycle')
    ax5.set_ylabel('Normalized Concentration')
    ax5.set_title('Normalized Total Mutant and WT Concentrations with Flanking Regions')
    ax5.legend(loc='upper center', fontsize=8, bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax5.grid(True, alpha=0.3)
    
    # Add figure title
    fig.suptitle("Figure Z-Flanking: Thermal Cycle Summary with Flanking Regions", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save or show
    if output_path:
        # Save as PNG
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
        
        # Save as PDF - replace extension with .pdf
        pdf_path = os.path.splitext(output_path)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Figure saved to {pdf_path}")
    else:
        plt.show()
    
    plt.close(fig)

def plot_mutation_distance_evolution(cycle_data, include_duplexes=False, output_path=None):
    """
    Generate a plot showing the evolution of the distance of alleles from the 3' end over Thermal Cycles.
    Shows the average distance for mutant and WT alleles.
    
    Args:
        cycle_data: Dictionary with cycle data
        include_duplexes: Whether to include oligos bound in duplexes; only affects plot title
        output_path: Path to save the figure
    """
    if not cycle_data['cycles']:
        print("No cycle data to plot")
        return
    
    cycles = cycle_data['cycles']
    data = cycle_data['data']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7.5, 5))
    
    # Prepare data for plotting
    mutant_avg_dist = []
    wt_avg_dist = []
    
    # Extract data for each cycle
    for i, cycle in enumerate(cycles):
        # Get weighted distance data (already weighted by concentration)
        mutant_dists = data[i]['mutant_3prime_dist_w']
        wt_dists = data[i]['wt_3prime_dist_w']
        
        # Calculate average distance (weighted by concentration)
        mutant_avg = np.mean(mutant_dists) if mutant_dists else np.nan
        wt_avg = np.mean(wt_dists) if wt_dists else np.nan
        
        mutant_avg_dist.append(mutant_avg)
        wt_avg_dist.append(wt_avg)
    
    # Plot average distance lines
    ax.plot(cycles, mutant_avg_dist, 'o-', color='#fbb040', linewidth=2, 
            label='Mutant Allele (Average)')
    ax.plot(cycles, wt_avg_dist, 'o-', color='#00a651', linewidth=2, 
            label='Wildtype Allele (Average)')
    
    # Set axis properties
    ax.set_xlabel('Thermal Cycle')
    ax.set_ylabel('Distance from 3\' End')
    ax.set_title(f'Evolution of Allele Distance from 3\' End Over Thermal Cycles\n'
                f'({("All Oligomers" if include_duplexes else "Single-Stranded Oligomers Only")})')
    
    # Set x-axis to show integer cycle numbers
    ax.set_xticks(cycles)
    ax.set_xticklabels([str(c) for c in cycles])
    
    # Add legend
    ax.legend(loc='best')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Save or show
    if output_path:
        # Save as PNG
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
        
        # Save as PDF - replace extension with .pdf
        pdf_path = os.path.splitext(output_path)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Figure saved to {pdf_path}")
    else:
        plt.show()
    
    plt.close(fig)

def plot_from_file(state_file, output_dir='plots', flanking_offset=1):
    """
    Generate all figures from the saved state file.
    
    Args:
        state_file: Path to the pickle file with saved state
        output_dir: Directory to save output plots
        flanking_offset: Number of positions to offset flanking region inward from oligo bounds (default: 1)
        
    Raises:
        ValueError: If the flanking_offset is too large and creates an invalid flanking region
    """
    # Check if file exists
    if not os.path.exists(state_file):
        print(f"Error: File {state_file} not found")
        return

    if not flanking_offset > -1:
        print(f"Error: Flanking offset must be non-negative")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the state
    print(f"Loading state from {state_file}...")
    state_data = ExtendedOligomerSystem.load_system_state(state_file)
    melted_data = melt_all_duplexes(state_data)
    
    # Generate base filename for outputs
    base_name = os.path.splitext(os.path.basename(state_file))[0]
    
    # Set flanking region based on the mutant-containing oligo
    flanking_region = None
    genome_length = state_data['genome_length']
    
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
    
    # Extract cycle data for all plots
    print("Extracting cycle data...")
    state_data['flanking_region'] = flanking_region
    melted_data['flanking_region'] = flanking_region
    cycle_data = extract_cycle_data_ss(state_data)
    melted_cycle_data = extract_cycle_data_ss(melted_data)
    
    # Generate all figures
    print("Generating Figure A (SS oligo counts)...")
    plot_length_histograms(
        cycle_data,
        include_duplexes=False,
        weighted=False,
        output_path=os.path.join(output_dir, f"{base_name}_FigureA.png")
    )
    
    print("Generating Figure B (SS oligo concentrations)...")
    plot_length_histograms(
        cycle_data,
        include_duplexes=False,
        weighted=True,
        output_path=os.path.join(output_dir, f"{base_name}_FigureB.png")
    )
    
    print("Generating Figure A-all (all oligo counts)...")
    plot_length_histograms(
        melted_cycle_data,
        include_duplexes=True,
        weighted=False,
        output_path=os.path.join(output_dir, f"{base_name}_FigureA-all.png")
    )
    
    print("Generating Figure B-all (all oligo concentrations)...")
    plot_length_histograms(
        melted_cycle_data,
        include_duplexes=True,
        weighted=True,
        output_path=os.path.join(output_dir, f"{base_name}_FigureB-all.png")
    )
    
    print("Generating Figure C (mutation distance from 3' end)...")
    plot_mutation_distance(
        cycle_data, 
        include_duplexes=False,
        output_path=os.path.join(output_dir, f"{base_name}_FigureC.png")
    )

    print("Generating Figure C-Heatmap (mutation distance from 3' end as heatmaps)...")
    plot_mutation_distance_heatmap(
        cycle_data, 
        include_duplexes=False,
        output_path=os.path.join(output_dir, f"{base_name}_FigureC-Heatmap.png")
    )

    print("Generating Figure C-all (mutation distance from 3' end)...")
    plot_mutation_distance(
        melted_cycle_data,
        include_duplexes=True,
        output_path=os.path.join(output_dir, f"{base_name}_FigureC-all.png")
    )
    
    print("Generating Figure C-all-Heatmap (mutation distance from 3' end as heatmaps)...")
    plot_mutation_distance_heatmap(
        melted_cycle_data,
        include_duplexes=True,
        output_path=os.path.join(output_dir, f"{base_name}_FigureC-all-Heatmap.png")
    )
    
    print("Generating Figure D (evolution of mutation distance from 3' end)...")
    plot_mutation_distance_evolution(
        cycle_data,
        include_duplexes=False,
        output_path=os.path.join(output_dir, f"{base_name}_FigureD.png")
    )
    
    print("Generating Figure Z (Thermal Cycle summary)...")
    plot_cycle_summary(
        state_data,
        cycle_data,
        melted_cycle_data,
        output_path=os.path.join(output_dir, f"{base_name}_FigureZ.png")
    )
    
    print("Generating Figure Z-Flanking (Thermal Cycle summary with flanking regions)...")
    plot_cycle_summary_flanking(
        state_data,
        cycle_data,
        melted_cycle_data,
        output_path=os.path.join(output_dir, f"{base_name}_FigureZ-Flanking.png")
    )
    
    print("Generating Figure C-FirstLast (first vs last cycle distance histograms)...")
    plot_mutation_distance_first_last(
        cycle_data,
        include_duplexes=False,
        output_path=os.path.join(output_dir, f"{base_name}_FigureC-FirstLast.png")
    )
    
    print("Generating Figure C-all-FirstLast (first vs last cycle distance histograms)...")
    plot_mutation_distance_first_last(
        melted_cycle_data,
        include_duplexes=True,
        output_path=os.path.join(output_dir, f"{base_name}_FigureC-all-FirstLast.png")
    )
    
    print("Done!")
    
def main():
    parser = argparse.ArgumentParser(description="Create advanced plots from saved system state")
    
    parser.add_argument('state_file', nargs='?', help="Path to the pickle file with saved state")
    parser.add_argument('--output-dir', default='plots', help="Directory to save output plots")
    parser.add_argument('--test-flanking-region', action='store_true', help="Run tests for contains_flanking_region function")
    
    args = parser.parse_args()
    
    if args.test_flanking_region:
        test_contains_flanking_region()
    elif args.state_file:
        plot_from_file(args.state_file, args.output_dir)
    else:
        parser.print_help()


def test_contains_flanking_region():
    """
    Unit tests for the contains_flanking_region function.
    """
    genome_length = 100
    
    # Case 1: Neither oligo nor flanking region wraps
    # Test 1A: Oligo fully contains flanking region (True)
    oligo1a = (10, 50, True, False, False)
    flanking1a = (20, 40)
    assert contains_flanking_region(oligo1a, flanking1a, genome_length) == True
    
    # Test 1B: Oligo partially contains flanking region (False)
    oligo1b = (30, 60, True, False, False)
    flanking1b = (20, 70)
    assert contains_flanking_region(oligo1b, flanking1b, genome_length) == False
    
    # Case 2: Flanking region wraps, oligo doesn't
    # Test 2A: Oligo doesn't wrap, can't contain wrapped flanking region (False)
    oligo2a = (10, 50, True, False, False)
    flanking2a = (80, 20)  # Wraps around from 80 to 20
    assert contains_flanking_region(oligo2a, flanking2a, genome_length) == False
    
    # Test 2B: Even a large non-wrapping oligo can't contain wrapped flanking (False)
    oligo2b = (5, 95, True, False, False)
    flanking2b = (90, 10)  # Wraps around from 90 to 10
    assert contains_flanking_region(oligo2b, flanking2b, genome_length) == False
    
    # Case 3: Oligo wraps, flanking region doesn't
    # Test 3A: Wrapped oligo contains flanking region (True)
    oligo3a = (80, 30, True, False, False)  # Wraps from 80 through 99, then 0 through 30
    flanking3a = (90, 20)
    assert contains_flanking_region(oligo3a, flanking3a, genome_length) == True
    
    # Test 3B: Wrapped oligo doesn't fully contain flanking region (False)
    oligo3b = (80, 10, True, False, False)  # Wraps from 80 through 99, then 0 through 10
    flanking3b = (85, 15)
    assert contains_flanking_region(oligo3b, flanking3b, genome_length) == False
    
    # Case 4: Both oligo and flanking region wrap
    # Test 4A: Wrapped oligo contains wrapped flanking region (True)
    oligo4a = (70, 30, True, False, False)  # Wraps from 70 through 99, then 0 through 30
    flanking4a = (80, 20)  # Wraps from 80 through 99, then 0 through 20
    assert contains_flanking_region(oligo4a, flanking4a, genome_length) == True
    
    # Test 4B: Wrapped oligo doesn't fully contain wrapped flanking region (False)
    oligo4b = (80, 30, True, False, False)  # Wraps from 80 through 99, then 0 through 30
    flanking4b = (70, 20)  # Wraps from 70 through 99, then 0 through 20
    assert contains_flanking_region(oligo4b, flanking4b, genome_length) == False
    
    print("All tests passed!")

def plot_mutation_distance_heatmap(cycle_data, include_duplexes=False, output_path=None):
    """
    Generate Figure C-Heatmap showing distribution of mutation distance from 3' end as heatmaps.
    Split into 6 panels for mutant CW, mutant CCW, wildtype CW, wildtype CCW,
    combined mutant (CW+CCW), and combined wildtype (CW+CCW) oligos.
    
    Args:
        cycle_data: Dictionary with cycle data
        include_duplexes: Whether to include oligos bound in duplexes; only affects plot title
        output_path: Path to save the figure
    """

    oligo_type_for_plot_title = "All Oligomers" if include_duplexes else "Single-Stranded Oligomers Only"

    if not cycle_data['cycles']:
        print("No cycle data to plot")
        return
    
    cycles = cycle_data['cycles']
    data = cycle_data['data']
    
    # Create figure with a 2x3 layout (6 panels) - make wider for square subplots
    fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(22, 12), sharey=True)
    
    # Define the data sources and corresponding axes
    plot_configs = [
        {
            'data_field': 'mutant_3prime_dist_w_cw',
            'conc_field': 'mutant_cw_conc',
            'axis': ax1,
            'title': "Mutant CW Oligos: Distance from 3' End to Mutation Site",
            'xlabel': "Thermal Cycle",
            'color': '#fbb040'
        },
        {
            'data_field': 'mutant_3prime_dist_w_ccw',
            'conc_field': 'mutant_ccw_conc',
            'axis': ax2,
            'title': "Mutant CCW Oligos: Distance from 3' End to Mutation Site",
            'xlabel': "Thermal Cycle",
            'color': '#fbb040'
        },
        {
            'data_field': 'wt_3prime_dist_w_cw',
            'conc_field': 'wt_cw_conc',
            'axis': ax3,
            'title': "Wildtype CW Oligos: Distance from 3' End to Wildtype Allele",
            'xlabel': "Thermal Cycle",
            'color': '#00a651'
        },
        {
            'data_field': 'wt_3prime_dist_w_ccw',
            'conc_field': 'wt_ccw_conc',
            'axis': ax4,
            'title': "Wildtype CCW Oligos: Distance from 3' End to Wildtype Allele",
            'xlabel': "Thermal Cycle",
            'color': '#00a651'
        },
        {
            'data_field': 'mutant_3prime_dist_w',
            'conc_field': 'mutant_conc',
            'axis': ax5,
            'title': "All Mutant Oligos: Distance from 3' End to Mutation Site",
            'xlabel': "Thermal Cycle",
            'color': '#fbb040'
        },
        {
            'data_field': 'wt_3prime_dist_w',
            'conc_field': 'wt_conc',
            'axis': ax6,
            'title': "All Wildtype Oligos: Distance from 3' End to Wildtype Allele",
            'xlabel': "Thermal Cycle",
            'color': '#00a651'
        }
    ]
    
    # First pass: determine global maximum distance across all subplots
    global_max_distance = 0
    for config in plot_configs:
        for i, cycle in enumerate(cycles):
            distances = data[i][config['data_field']]
            if distances:
                max_dist = max(distances)
                global_max_distance = max(global_max_distance, max_dist)
    
    # Set a minimum range if no data found
    if global_max_distance == 0:
        global_max_distance = 10
    
    # Create distance bins of size 5
    bin_size = 5
    bin_edges = np.arange(0, global_max_distance + bin_size, bin_size)
    if bin_edges[-1] < global_max_distance:
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin_size)
    
    # Create bin labels for y-axis
    bin_labels = [f"{bin_edges[i]}" for i in range(len(bin_edges) - 1)]
    
    # Process each subplot
    for config in plot_configs:
        # Prepare heatmap data
        # Create matrix where rows are distance bins and columns are cycles
        heatmap_data = np.zeros((len(bin_edges) - 1, len(cycles)))
        
        for i, cycle in enumerate(cycles):
            distances = data[i][config['data_field']]
            if not distances:
                continue
                
            # Create a frequency histogram for each cycle using the defined bins
            if len(distances) > 0:
                hist, _ = np.histogram(distances, bins=bin_edges)
                
                # Normalize histogram so that max height is 1.0 (same as ridge plot)
                if np.sum(hist) > 0:
                    hist = hist / np.max(hist)
                
                # Store normalized histogram data
                heatmap_data[:, i] = hist
        
        # Create allele-specific colormap (white to allele color)
        if 'mutant' in config['data_field']:
            # Mutant: white to orange
            cmap = LinearSegmentedColormap.from_list('mutant_cmap', ['#FFFFFF', config['color']])
        else:
            # Wildtype: white to green
            cmap = LinearSegmentedColormap.from_list('wt_cmap', ['#FFFFFF', config['color']])
        
        # Create heatmap
        sns.heatmap(
            heatmap_data,
            cmap=cmap,
            ax=config['axis'],
            cbar_kws={'label': 'Normalized Frequency'},
            xticklabels=[f"{c}" for c in cycles],
            yticklabels=bin_labels,
            vmin=0,
            vmax=1
        )
        
        # Flip y-axis so increasing distance goes upward
        config['axis'].invert_yaxis()
        
        # Set labels and title
        config['axis'].set_xlabel(config['xlabel'])
        config['axis'].set_ylabel("Distance from 3' End")
        config['axis'].set_title(config['title'] + " (" + oligo_type_for_plot_title + ")")
        
        # Explicitly set y-tick labels to ensure they're visible on all subplots
        config['axis'].set_yticks(range(len(bin_labels)))
        config['axis'].set_yticklabels(bin_labels, rotation=0)
        
        # Make sure tick marks are visible on both axes
        config['axis'].tick_params(axis='both', which='major', length=6, width=1, direction='out')
        config['axis'].tick_params(axis='x', which='major', top=False, bottom=True)
        config['axis'].tick_params(axis='y', which='major', left=True, right=False)
        
        # Ensure y-axis labels are visible even with sharey=True
        config['axis'].tick_params(labelleft=True)
    
    # Add figure title
    fig.suptitle("Heatmap of Distance from 3' End to Mutation/Wildtype Site by Orientation (" + oligo_type_for_plot_title + ")", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save or show
    if output_path:
        # Save as PNG
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
        
        # Save as PDF - replace extension with .pdf
        pdf_path = os.path.splitext(output_path)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Figure saved to {pdf_path}")
    else:
        plt.show()
    
    plt.close(fig)

def plot_mutation_distance_first_last(cycle_data, include_duplexes=False, output_path=None):
    """
    Generate Figure C-FirstLast showing first and last distance distributions as separate histograms.
    Creates two separate figures: one for first cycle, one for last cycle.
    Split into 6 panels for mutant CW, mutant CCW, wildtype CW, wildtype CCW,
    combined mutant (CW+CCW), and combined wildtype (CW+CCW) oligos.
    
    Args:
        cycle_data: Dictionary with cycle data
        include_duplexes: Whether to include oligos bound in duplexes; only affects plot title
        output_path: Path to save the figure (will create two files: *_First.png and *_Last.png)
    """

    oligo_type_for_plot_title = "All Oligomers" if include_duplexes else "Single-Stranded Oligomers Only"

    if not cycle_data['cycles']:
        print("No cycle data to plot")
        return
    
    cycles = cycle_data['cycles']
    data = cycle_data['data']
    
    if len(cycles) < 2:
        print("Need at least 2 cycles to show first and last distributions")
        return
    
    # Get first and last cycle indices
    first_cycle_idx = 0
    last_cycle_idx = len(cycles) - 1
    first_cycle = cycles[first_cycle_idx]
    last_cycle = cycles[last_cycle_idx]
    
    # Define the data sources and corresponding axes
    plot_configs = [
        {
            'data_field': 'mutant_3prime_dist_w_cw',
            'conc_field': 'mutant_cw_conc',
            'title': "Mutant CW Oligos: Distance from 3' End to Mutation Site",
            'xlabel': "Distance from 3' End to Mutation",
            'color': '#fbb040'
        },
        {
            'data_field': 'mutant_3prime_dist_w_ccw',
            'conc_field': 'mutant_ccw_conc',
            'title': "Mutant CCW Oligos: Distance from 3' End to Mutation Site",
            'xlabel': "Distance from 3' End to Mutation",
            'color': '#fbb040'
        },
        {
            'data_field': 'wt_3prime_dist_w_cw',
            'conc_field': 'wt_cw_conc',
            'title': "Wildtype CW Oligos: Distance from 3' End to Wildtype Allele",
            'xlabel': "Distance from 3' End to Wildtype Allele",
            'color': '#00a651'
        },
        {
            'data_field': 'wt_3prime_dist_w_ccw',
            'conc_field': 'wt_ccw_conc',
            'title': "Wildtype CCW Oligos: Distance from 3' End to Wildtype Allele",
            'xlabel': "Distance from 3' End to Wildtype Allele",
            'color': '#00a651'
        },
        {
            'data_field': 'mutant_3prime_dist_w',
            'conc_field': 'mutant_conc',
            'title': "All Mutant Oligos: Distance from 3' End to Mutation Site",
            'xlabel': "Distance from 3' End to Mutation",
            'color': '#fbb040'
        },
        {
            'data_field': 'wt_3prime_dist_w',
            'conc_field': 'wt_conc',
            'title': "All Wildtype Oligos: Distance from 3' End to Wildtype Allele",
            'xlabel': "Distance from 3' End to Wildtype Allele",
            'color': '#00a651'
        }
    ]
    
    # First pass: determine global maximum distance across all subplots
    global_max_distance = 0
    for config in plot_configs:
        for cycle_idx in [first_cycle_idx, last_cycle_idx]:
            distances = data[cycle_idx][config['data_field']]
            if distances:
                max_dist = max(distances)
                global_max_distance = max(global_max_distance, max_dist)
    
    # Set a minimum range if no data found
    if global_max_distance == 0:
        global_max_distance = 10
    
    # Create bin edges
    bin_edges = np.arange(0, global_max_distance + 2)  # +2 to include max value
    
    # Function to create a single figure for a given cycle
    def create_cycle_figure(cycle_idx, cycle_num, suffix):
        # Create figure with a 2x3 layout (6 panels)
        fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(9, 6))
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        
        # Process each subplot
        for i, config in enumerate(plot_configs):
            ax = axes[i]
            
            # Get data for this cycle
            distances = data[cycle_idx][config['data_field']]
            
            # Create histogram if data exists
            if distances:
                hist, _ = np.histogram(distances, bins=bin_edges)
                # Normalize so max height is 1.0
                if np.max(hist) > 0:
                    hist = hist / np.max(hist)
                
                # Plot as bars
                ax.bar(bin_edges[:-1], hist, width=0.8, alpha=0.8, 
                       color=config['color'], 
                       edgecolor='none', linewidth=0)
            
            # Set axis properties
            ax.set_xlim(-0.5, global_max_distance + 0.5)
            ax.set_ylim(0, 1.1)  # Normalized frequency from 0 to 1
            ax.set_xlabel(config['xlabel'])
            ax.set_ylabel('Normalized Frequency')
            ax.set_title(config['title'] + " (" + oligo_type_for_plot_title + ")")
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3)
            
            # Make sure tick marks are visible
            ax.tick_params(axis='both', which='major', length=6, width=1, direction='out')
        
        # Add figure title
        fig.suptitle(f"Cycle {cycle_num} Distance Distributions (" + oligo_type_for_plot_title + ")", fontsize=14, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save figure
        if output_path:
            # Create filename with suffix
            base_path = os.path.splitext(output_path)[0]
            cycle_output_path = f"{base_path}_{suffix}.png"
            
            # Save as PNG
            plt.savefig(cycle_output_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {cycle_output_path}")
            
            # Save as PDF
            pdf_path = os.path.splitext(cycle_output_path)[0] + '.pdf'
            plt.savefig(pdf_path, bbox_inches='tight')
            print(f"Figure saved to {pdf_path}")
        else:
            plt.show()
        
        plt.close(fig)
    
    # Create separate figures for first and last cycles
    create_cycle_figure(first_cycle_idx, first_cycle, "First")
    create_cycle_figure(last_cycle_idx, last_cycle, "Last")

if __name__ == "__main__":
    main() 