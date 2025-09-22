from vcg_extended_fns import ExtendedOligomerSystem
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
import re
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

def load_system_state(filename):
    """
    Load the system state from a pickle file
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def can_anneal(oligo1, oligo2, min_overlap=2, genome_length=100):
    """
    Check if two oligos can anneal, based on the CoreOligomerSystem.can_anneal logic
    """
    # Extract the relevant parts
    start1, end1, is_clockwise1, is_duplex1, _ = oligo1[:5]
    start2, end2, is_clockwise2, is_duplex2, _ = oligo2[:5]
    
    # Early rejection tests
    # Duplexes cannot anneal again
    if is_duplex1 or is_duplex2:
        return False, 0
    
    # Oligomers must have opposite directions to anneal
    if is_clockwise1 == is_clockwise2:
        return False, 0
    
    # Calculate overlap using bit arrays
    bit_array1 = np.zeros(genome_length, dtype=bool)
    bit_array2 = np.zeros(genome_length, dtype=bool)
    
    # Fill positions for first oligo
    if end1 >= start1:
        bit_array1[start1:end1+1] = True
    else:  # Wraps around
        bit_array1[start1:] = True
        bit_array1[:end1+1] = True
    
    # Fill positions for second oligo
    if end2 >= start2:
        bit_array2[start2:end2+1] = True
    else:  # Wraps around
        bit_array2[start2:] = True
        bit_array2[:end2+1] = True
    
    # Find overlap
    overlap = bit_array1 & bit_array2
    overlap_size = np.sum(overlap)
    
    # Return result
    return overlap_size >= min_overlap, overlap_size

def calculate_oligo_3prime_end(oligo):
    """
    Determine the 3' end position of an oligo based on its direction
    """
    start, end, is_clockwise, _, _ = oligo[:5]
    
    # For clockwise oligos, 3' end is at 'end'
    # For counterclockwise oligos, 3' end is at 'start'
    return end if is_clockwise else start

def calculate_mutation_distance_from_3prime(oligo, mutation_site, genome_length):
    """
    Calculate the distance from the mutation site to the 3' end of the oligo
    """
    start, end, is_clockwise, _, is_mutant = oligo[:5]
    
    # Skip non-mutant oligos
    if not is_mutant:
        return None
    
    # Get 3' end position
    prime3_pos = calculate_oligo_3prime_end(oligo)
    
    # Calculate distance considering circular genome
    if is_clockwise:
        # For clockwise, we move backward from 3' (at end)
        if prime3_pos >= mutation_site:
            return prime3_pos - mutation_site
        else:
            return prime3_pos + genome_length - mutation_site
    else:
        # For counterclockwise, we move forward from 3' (at start)
        if mutation_site >= prime3_pos:
            return mutation_site - prime3_pos
        else:
            return genome_length - prime3_pos + mutation_site

def calculate_wt_distance_from_3prime(oligo, mutation_site, genome_length):
    """
    Calculate the distance from the mutation site to the 3' end of the oligo 
    for wild-type (non-mutant) oligos, but only if the oligo contains the mutation site.
    """
    start, end, is_clockwise, _, is_mutant = oligo[:5]
    
    # Skip mutant oligos
    if is_mutant:
        return None
    
    # Check if the oligo segment contains the mutation site
    contains_mutation = False
    if end >= start:  # Normal segment
        contains_mutation = start <= mutation_site <= end
    else:  # Segment that wraps around the circular genome
        contains_mutation = (mutation_site >= start) or (mutation_site <= end)
    
    # If the oligo doesn't contain the mutation site, return None
    if not contains_mutation:
        return None
    
    # Get 3' end position
    prime3_pos = calculate_oligo_3prime_end(oligo)
    
    # Calculate distance considering circular genome
    if is_clockwise:
        # For clockwise, we move backward from 3' (at end)
        if prime3_pos >= mutation_site:
            return prime3_pos - mutation_site
        else:
            return prime3_pos + genome_length - mutation_site
    else:
        # For counterclockwise, we move forward from 3' (at start)
        if mutation_site >= prime3_pos:
            return mutation_site - prime3_pos
        else:
            return genome_length - prime3_pos + mutation_site

def oligo_contains_mutation_site(oligo, mutation_site, genome_length):
    """
    Check if an oligo contains the mutation site within its sequence.
    """
    start, end, is_clockwise, _, _ = oligo[:5]
    
    # Check if the oligo segment contains the mutation site
    if end >= start:  # Normal segment
        return start <= mutation_site <= end
    else:  # Segment that wraps around the circular genome
        return (mutation_site >= start) or (mutation_site <= end)

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

def analyze_pcr_cycles(state_data, min_overlap=3):
    """
    Analyze PCR cycle data to evaluate mutant oligo binding partners
    
    Args:
        state_data: Loaded state data dictionary
        min_overlap: Minimum overlap required for annealing (default: 3)
    """
    genome_length = state_data['genome_length']
    mutation_site = genome_length // 2
    cycle_states = state_data['cycle_states']
    
    # Data to collect
    mutant_analysis = []
    
    # Go through each cycle
    print(f"Analyzing {len(cycle_states)} PCR cycles...")
    for cycle_idx, cycle_data in enumerate(cycle_states):
        if cycle_idx % 2 == 0:
            print(f"Processing cycle {cycle_idx}/{len(cycle_states)}...")
            
        cycle_oligos = cycle_data.get('oligomers', [])
        cycle_concs = cycle_data.get('concentrations', [])
        
        if len(cycle_oligos) == 0 or len(cycle_concs) == 0:
            continue
        
        # Get single stranded oligos
        ss_oligos = []
        for i, oligo in enumerate(cycle_oligos):
            if i < len(cycle_concs) and not oligo[3] and cycle_concs[i] > 0:  # Not a duplex and positive concentration
                ss_oligos.append((oligo, cycle_concs[i]))
        
        # Identify mutant oligos
        mutant_ss_oligos = [(oligo, conc) for oligo, conc in ss_oligos if oligo[4] > 0]
        
        # Identify wild-type oligos that contain the mutation site
        wt_ss_oligos_with_site = []
        for oligo, conc in ss_oligos:
            if oligo[4] == 0:  # Non-mutant oligo
                wt_distance = calculate_wt_distance_from_3prime(oligo, mutation_site, genome_length)
                if wt_distance is not None:  # Contains the mutation site
                    wt_ss_oligos_with_site.append((oligo, conc))
        
        # Process each mutant oligo
        for mutant_oligo, mutant_conc in mutant_ss_oligos:
            # Find all potential binding partners
            binding_partners = []
            for partner_oligo, partner_conc in ss_oligos:
                # Skip self-comparison
                if mutant_oligo == partner_oligo:
                    continue
                    
                # Check if can anneal
                can_bind, overlap = can_anneal(mutant_oligo, partner_oligo, min_overlap=min_overlap, genome_length=genome_length)
                if can_bind:
                    binding_partners.append((partner_oligo, partner_conc, overlap))
            
            if not binding_partners:
                continue
                
            # Get mutation position
            mutant_3prime = calculate_oligo_3prime_end(mutant_oligo)
            mutation_distance = calculate_mutation_distance_from_3prime(mutant_oligo, mutation_site, genome_length)
            
            # Classify binding partners into productive and non-productive
            productive_partners = []
            non_productive_partners = []
            
            for partner_oligo, partner_conc, overlap in binding_partners:
                partner_3prime = calculate_oligo_3prime_end(partner_oligo)
                
                # Determine if this binding partner will pick up the mutation
                # For a clockwise mutant oligo, binding partners whose 3' end is between mutation site and mutant's 3' end
                # For a counterclockwise mutant oligo, binding partners whose 3' end is between mutation site and mutant's 3' end
                is_productive = False
                
                mutant_start, mutant_end, mutant_is_clockwise, _, _ = mutant_oligo[:5]
                
                if mutant_is_clockwise:
                    # For clockwise mutant, check if partner's 3' end position is between mutation_site and end
                    if mutant_end >= mutation_site:
                        # No wrap-around
                        is_productive = mutation_site <= partner_3prime <= mutant_end
                    else:
                        # Wrap-around
                        is_productive = partner_3prime >= mutation_site or partner_3prime <= mutant_end
                else:
                    # For counterclockwise mutant, check if partner's 3' end position is between start and mutation_site
                    if mutant_start <= mutation_site:
                        # No wrap-around
                        is_productive = mutant_start <= partner_3prime <= mutation_site
                    else:
                        # Wrap-around
                        is_productive = partner_3prime >= mutant_start or partner_3prime <= mutation_site
                
                # Additional check: binding partner must not already contain the mutation site
                if is_productive:
                    is_productive = not oligo_contains_mutation_site(partner_oligo, mutation_site, genome_length)
                
                if is_productive:
                    productive_partners.append((partner_oligo, partner_conc))
                else:
                    non_productive_partners.append((partner_oligo, partner_conc))
            
            # Calculate total concentrations
            total_productive_conc = sum([conc for _, conc in productive_partners])
            total_non_productive_conc = sum([conc for _, conc in non_productive_partners])
            total_binding_conc = total_productive_conc + total_non_productive_conc
            
            # Calculate ratio
            ratio = 0 if total_binding_conc == 0 else total_productive_conc / total_binding_conc
            
            # Store the results
            mutant_analysis.append({
                'cycle': cycle_idx,
                'oligo_type': 'mutant',
                'oligo': mutant_oligo,
                'mutation_distance': mutation_distance,
                'conc': mutant_conc,
                'total_binding_partners': len(binding_partners),
                'productive_partners': len(productive_partners),
                'non_productive_partners': len(non_productive_partners),
                'productive_conc': total_productive_conc,
                'non_productive_conc': total_non_productive_conc,
                'ratio': ratio
            })
        
        # Process each wild-type oligo that contains the mutation site
        for wt_oligo, wt_conc in wt_ss_oligos_with_site:
            # Find all potential binding partners
            binding_partners = []
            for partner_oligo, partner_conc in ss_oligos:
                # Skip self-comparison
                if wt_oligo == partner_oligo:
                    continue
                    
                # Check if can anneal
                can_bind, overlap = can_anneal(wt_oligo, partner_oligo, min_overlap=min_overlap, genome_length=genome_length)
                if can_bind:
                    binding_partners.append((partner_oligo, partner_conc, overlap))
            
            if not binding_partners:
                continue
                
            # Get mutation position
            wt_3prime = calculate_oligo_3prime_end(wt_oligo)
            mutation_distance = calculate_wt_distance_from_3prime(wt_oligo, mutation_site, genome_length)
            
            # Classify binding partners into productive and non-productive
            productive_partners = []
            non_productive_partners = []
            
            for partner_oligo, partner_conc, overlap in binding_partners:
                partner_3prime = calculate_oligo_3prime_end(partner_oligo)
                
                # Determine if this binding partner will pick up the wild-type
                # Same logic as for mutants, but for wild-type oligos
                is_productive = False
                
                wt_start, wt_end, wt_is_clockwise, _, _ = wt_oligo[:5]
                
                if wt_is_clockwise:
                    # For clockwise wild-type, check if partner's 3' end position is between mutation_site and end
                    if wt_end >= mutation_site:
                        # No wrap-around
                        is_productive = mutation_site <= partner_3prime <= wt_end
                    else:
                        # Wrap-around
                        is_productive = partner_3prime >= mutation_site or partner_3prime <= wt_end
                else:
                    # For counterclockwise wild-type, check if partner's 3' end position is between start and mutation_site
                    if wt_start <= mutation_site:
                        # No wrap-around
                        is_productive = wt_start <= partner_3prime <= mutation_site
                    else:
                        # Wrap-around
                        is_productive = partner_3prime >= wt_start or partner_3prime <= mutation_site
                
                # Additional check: binding partner must not already contain the mutation site
                if is_productive:
                    is_productive = not oligo_contains_mutation_site(partner_oligo, mutation_site, genome_length)
                
                if is_productive:
                    productive_partners.append((partner_oligo, partner_conc))
                else:
                    non_productive_partners.append((partner_oligo, partner_conc))
            
            # Calculate total concentrations
            total_productive_conc = sum([conc for _, conc in productive_partners])
            total_non_productive_conc = sum([conc for _, conc in non_productive_partners])
            total_binding_conc = total_productive_conc + total_non_productive_conc
            
            # Calculate ratio
            ratio = 0 if total_binding_conc == 0 else total_productive_conc / total_binding_conc
            
            # Store the results
            mutant_analysis.append({
                'cycle': cycle_idx,
                'oligo_type': 'wildtype',
                'oligo': wt_oligo,
                'mutation_distance': mutation_distance,
                'conc': wt_conc,
                'total_binding_partners': len(binding_partners),
                'productive_partners': len(productive_partners),
                'non_productive_partners': len(non_productive_partners),
                'productive_conc': total_productive_conc,
                'non_productive_conc': total_non_productive_conc,
                'ratio': ratio
            })
    
    return mutant_analysis

def plot_mutant_scatter(mutant_analysis, last_cycle, output_dir='.', prefix=''):
    """
    Create scatter plots of mutant oligos showing ratio vs distance for the last cycle
    """
    last_cycle_data = [data for data in mutant_analysis if data['cycle'] == last_cycle]
    
    # Split data based on direction
    mutant_cw_data = [data for data in last_cycle_data 
               if data['mutation_distance'] is not None and data['oligo'][2] and data['oligo_type'] == 'mutant']
    mutant_ccw_data = [data for data in last_cycle_data 
                if data['mutation_distance'] is not None and not data['oligo'][2] and data['oligo_type'] == 'mutant']
    
    # Create a figure with two subplots for mutant oligos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot mutant CW data on first subplot
    if mutant_cw_data:
        distances = [data['mutation_distance'] for data in mutant_cw_data]
        ratios = [data['ratio'] for data in mutant_cw_data]
        concs = [data['conc'] for data in mutant_cw_data]
        
        scatter1 = ax1.scatter(distances, ratios, c=concs, cmap='viridis', alpha=0.7, s=50)
        fig.colorbar(scatter1, ax=ax1, label='Oligo Concentration')
    
    ax1.set_xlabel('Distance of Mutation from 3\' End')
    ax1.set_ylabel('Productive / (Productive + Non-productive) Ratio')
    ax1.set_title(f'Clockwise Mutant Oligos - Cycle {last_cycle}')
    ax1.grid(alpha=0.3)
    
    # Plot mutant CCW data on second subplot
    if mutant_ccw_data:
        distances = [data['mutation_distance'] for data in mutant_ccw_data]
        ratios = [data['ratio'] for data in mutant_ccw_data]
        concs = [data['conc'] for data in mutant_ccw_data]
        
        scatter2 = ax2.scatter(distances, ratios, c=concs, cmap='viridis', alpha=0.7, s=50)
        fig.colorbar(scatter2, ax=ax2, label='Oligo Concentration')
    
    ax2.set_xlabel('Distance of Mutation from 3\' End')
    ax2.set_ylabel('Productive / (Productive + Non-productive) Ratio')
    ax2.set_title(f'Counterclockwise Mutant Oligos - Cycle {last_cycle}')
    ax2.grid(alpha=0.3)
    
    # Save mutant plot
    plt.tight_layout()
    png_path = os.path.join(output_dir, f'{prefix}mutant_cw_ccw_ratio_cycle_{last_cycle}.png')
    pdf_path = os.path.join(output_dir, f'{prefix}mutant_cw_ccw_ratio_cycle_{last_cycle}.pdf')
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()


def plot_wildtype_scatter(mutant_analysis, last_cycle, output_dir='.', prefix=''):
    """
    Create scatter plots of wild-type oligos showing ratio vs distance for the last cycle
    """
    last_cycle_data = [data for data in mutant_analysis if data['cycle'] == last_cycle]
    
    # Split data based on direction
    wt_cw_data = [data for data in last_cycle_data 
               if data['mutation_distance'] is not None and data['oligo'][2] and data['oligo_type'] == 'wildtype']
    wt_ccw_data = [data for data in last_cycle_data 
                if data['mutation_distance'] is not None and not data['oligo'][2] and data['oligo_type'] == 'wildtype']
    
    # Create a figure with two subplots for wild-type oligos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot wild-type CW data on first subplot
    if wt_cw_data:
        distances = [data['mutation_distance'] for data in wt_cw_data]
        ratios = [data['ratio'] for data in wt_cw_data]
        concs = [data['conc'] for data in wt_cw_data]
        
        scatter1 = ax1.scatter(distances, ratios, c=concs, cmap='viridis', alpha=0.7, s=50)
        fig.colorbar(scatter1, ax=ax1, label='Oligo Concentration')
    
    ax1.set_xlabel('Distance of Mutation from 3\' End')
    ax1.set_ylabel('Productive / (Productive + Non-productive) Ratio')
    ax1.set_title(f'Clockwise Wild-Type Oligos - Cycle {last_cycle}')
    ax1.grid(alpha=0.3)
    
    # Plot wild-type CCW data on second subplot
    if wt_ccw_data:
        distances = [data['mutation_distance'] for data in wt_ccw_data]
        ratios = [data['ratio'] for data in wt_ccw_data]
        concs = [data['conc'] for data in wt_ccw_data]
        
        scatter2 = ax2.scatter(distances, ratios, c=concs, cmap='viridis', alpha=0.7, s=50)
        fig.colorbar(scatter2, ax=ax2, label='Oligo Concentration')
    
    ax2.set_xlabel('Distance of Mutation from 3\' End')
    ax2.set_ylabel('Productive / (Productive + Non-productive) Ratio')
    ax2.set_title(f'Counterclockwise Wild-Type Oligos - Cycle {last_cycle}')
    ax2.grid(alpha=0.3)
    
    # Save wild-type plot
    plt.tight_layout()
    png_path = os.path.join(output_dir, f'{prefix}wildtype_cw_ccw_ratio_cycle_{last_cycle}.png')
    pdf_path = os.path.join(output_dir, f'{prefix}wildtype_cw_ccw_ratio_cycle_{last_cycle}.pdf')
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()


def plot_combined_avg_ratio(mutant_analysis, output_dir='.', prefix=''):
    """
    Create a combined figure showing average ratio vs distance for all oligo types
    with shared y-axis
    """
    # Get all available cycles and distances
    all_cycles = sorted(set([data['cycle'] for data in mutant_analysis]))
    all_distances = sorted(set([data['mutation_distance'] for data in mutant_analysis 
                               if data['mutation_distance'] is not None]))
    
    # Set up a colormap for cycles
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / len(all_cycles)) for i in range(len(all_cycles))]
    
    # Create a figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=True)
    
    # Define the subplot positions
    subplot_config = {
        'mutant': {
            True: {'title': 'Clockwise Mutant Oligos', 'ax': axes[0, 0]},  # Clockwise Mutant
            False: {'title': 'Counterclockwise Mutant Oligos', 'ax': axes[0, 1]}  # Counterclockwise Mutant
        },
        'wildtype': {
            True: {'title': 'Clockwise Wild-Type Oligos', 'ax': axes[1, 0]},  # Clockwise Wild-type
            False: {'title': 'Counterclockwise Wild-Type Oligos', 'ax': axes[1, 1]}  # Counterclockwise Wild-type
        }
    }
    
    # Track min and max ratio values for y-axis scaling
    min_ratio = float('inf')
    max_ratio = float('-inf')
    
    # Generate data for all four subplots
    for oligo_type in ['mutant', 'wildtype']:
        for is_clockwise in [True, False]:
            ax = subplot_config[oligo_type][is_clockwise]['ax']
            
            # Plot data for each cycle
            for i, cycle in enumerate(all_cycles):
                # Get data for this cycle and oligo type
                cycle_data = [data for data in mutant_analysis 
                             if data['cycle'] == cycle and 
                             data['oligo_type'] == oligo_type]
                
                # Filter by direction
                direction_data = [data for data in cycle_data 
                                 if data['mutation_distance'] is not None and 
                                 (data['oligo'][2] == is_clockwise)]
                
                if direction_data:
                    # Group by distance and calculate average ratio (simple average)
                    avg_ratios = {}
                    for distance in all_distances:
                        distance_data = [data for data in direction_data 
                                        if data['mutation_distance'] == distance]
                        if distance_data:
                            avg_ratios[distance] = sum([data['ratio'] for data in distance_data]) / len(distance_data)
                    
                    if avg_ratios:
                        distances = sorted(avg_ratios.keys())
                        ratios = [avg_ratios[d] for d in distances]
                        
                        # Update min/max for y-axis scaling
                        if ratios:
                            min_ratio = min(min_ratio, min(ratios))
                            max_ratio = max(max_ratio, max(ratios))
                        
                        ax.plot(distances, ratios, marker='o', linestyle='-', color=colors[i], 
                                label=f'Cycle {cycle}')
            
            # Configure subplot
            ax.set_xlabel('Distance of Mutation from 3\' End')
            ax.set_ylabel('Average Productive Ratio')
            ax.set_title(subplot_config[oligo_type][is_clockwise]['title'] + ' - Average Ratio vs Distance')
            ax.grid(alpha=0.3)
            
            # Set up legend without showing it yet
            if len(all_cycles) > 10:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.legend()
    
    # Ensure all subplots have the same y-axis limits with a bit of padding
    y_padding = (max_ratio - min_ratio) * 0.05  # 5% padding
    for ax_row in axes:
        for ax in ax_row:
            ax.set_ylim(max(0, min_ratio - y_padding), max_ratio + y_padding)
    
    # Add a single legend for all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=min(len(all_cycles), 5))
        # Remove individual legends
        for ax_row in axes:
            for ax in ax_row:
                ax.get_legend().remove()
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.98])  # Make room for the common legend
    png_path = os.path.join(output_dir, f'{prefix}combined_avg_ratio_vs_distance_per_cycle.png')
    pdf_path = os.path.join(output_dir, f'{prefix}combined_avg_ratio_vs_distance_per_cycle.pdf')
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()


def plot_weighted_ratio_summary(mutant_analysis, output_dir='.', prefix=''):
    """
    Create a summary plot showing weighted productive ratio across cycles for each oligo type
    """
    # Get all available cycles
    all_cycles = sorted(set([data['cycle'] for data in mutant_analysis]))
    
    plt.figure(figsize=(12, 8))
    
    # Define the four oligo types we want to track
    oligo_types = [
        {'type': 'mutant', 'is_clockwise': True, 'label': 'Clockwise Mutant', 'color': '#fbb040', 'marker': 'o', 'linestyle': '-'},
        {'type': 'mutant', 'is_clockwise': False, 'label': 'Counterclockwise Mutant', 'color': '#fbb040', 'marker': 'o', 'linestyle': '--'},
        {'type': 'wildtype', 'is_clockwise': True, 'label': 'Clockwise Wild-Type', 'color': '#00a651', 'marker': 'o', 'linestyle': '-'},
        {'type': 'wildtype', 'is_clockwise': False, 'label': 'Counterclockwise Wild-Type', 'color': '#00a651', 'marker': 'o', 'linestyle': '--'}
    ]
    
    # Calculate weighted sum (ratio * concentration) for each oligo type in each cycle
    cycle_values = {}
    for cycle in all_cycles:
        cycle_values[cycle] = {}
        cycle_data = [data for data in mutant_analysis if data['cycle'] == cycle]
        
        for oligo_type in oligo_types:
            # Filter data for this oligo type
            type_data = [data for data in cycle_data 
                       if data['oligo_type'] == oligo_type['type'] and 
                       data['oligo'][2] == oligo_type['is_clockwise'] and
                       data['mutation_distance'] is not None]
            
            # Calculate weighted average
            if type_data and sum(data['conc'] for data in type_data) > 0:
                weighted_avg = sum(data['ratio'] * data['conc'] for data in type_data) / sum(data['conc'] for data in type_data)
            else:
                weighted_avg = np.nan
            cycle_values[cycle][oligo_type['label']] = weighted_avg

    # Plot the values for each oligo type across cycles
    for oligo_type in oligo_types:
        label = oligo_type['label']
        values = [cycle_values[cycle].get(label, 0) for cycle in all_cycles]
        plt.plot(all_cycles, values, 
                 marker=oligo_type['marker'], 
                 color=oligo_type['color'], 
                 label=label, 
                 linestyle=oligo_type['linestyle'])
    
    plt.xlabel('PCR Cycle')
    plt.ylabel('Weighted Average Productive Ratio')
    plt.title('Weighted Productive Ratio by Oligo Type Across Cycles')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(output_dir, f'{prefix}weighted_ratio_summary_by_oligo_type.png')
    pdf_path = os.path.join(output_dir, f'{prefix}weighted_ratio_summary_by_oligo_type.pdf')
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

def plot_duplex_weighted_ratio_summary(mutant_analysis, melted_mutant_analysis, output_dir='.', prefix=''):
    """
    Create a summary plot showing weighted productive ratio across cycles for each oligo type
    """
    # Get all available cycles
    all_cycles = sorted(set([data['cycle'] for data in mutant_analysis]))
    
    plt.figure(figsize=(7.5, 5))
    
    # Define the four oligo types we want to track
    oligo_types = [
        {'type': 'mutant', 'is_clockwise': True, 'label': 'Clockwise Mutant', 'color': '#fbb040', 'marker': 'o', 'linestyle': '-'},
        {'type': 'mutant', 'is_clockwise': False, 'label': 'Counterclockwise Mutant', 'color': '#fbb040', 'marker': 'o', 'linestyle': '--'},
        {'type': 'wildtype', 'is_clockwise': True, 'label': 'Clockwise Wild-Type', 'color': '#00a651', 'marker': 'o', 'linestyle': '-'},
        {'type': 'wildtype', 'is_clockwise': False, 'label': 'Counterclockwise Wild-Type', 'color': '#00a651', 'marker': 'o', 'linestyle': '--'},
        {'type': 'mutant', 'is_clockwise': None, 'label': 'All Mutant', 'color': '#fbb040', 'marker': 's', 'linestyle': '-'},
        {'type': 'wildtype', 'is_clockwise': None, 'label': 'All Wild-Type', 'color': '#00a651', 'marker': 's', 'linestyle': '-'}
    ]
    
    # Calculate weighted sum (ratio * concentration) for each oligo type in each cycle
    cycle_values = {}
    melted_cycle_values = {}
    for cycle in all_cycles:
        cycle_values[cycle] = {}
        cycle_data = [data for data in mutant_analysis if data['cycle'] == cycle]

        melted_cycle_values[cycle] = {}
        melted_cycle_data = [data for data in melted_mutant_analysis if data['cycle'] == cycle]

        for oligo_type in oligo_types:
            # Filter data for this oligo type
            if oligo_type['is_clockwise'] is None:
                # All oligos of a given type (mutant or wildtype), regardless of direction
                type_data = [data for data in cycle_data 
                          if data['oligo_type'] == oligo_type['type'] and
                          data['mutation_distance'] is not None]
                
                melted_type_data = [data for data in melted_cycle_data 
                                 if data['oligo_type'] == oligo_type['type'] and
                                 data['mutation_distance'] is not None]
            else:
                # Specific oligo type and direction
                type_data = [data for data in cycle_data 
                          if data['oligo_type'] == oligo_type['type'] and 
                          data['oligo'][2] == oligo_type['is_clockwise'] and
                          data['mutation_distance'] is not None]
                
                melted_type_data = [data for data in melted_cycle_data 
                                 if data['oligo_type'] == oligo_type['type'] and 
                                 data['oligo'][2] == oligo_type['is_clockwise'] and
                                 data['mutation_distance'] is not None]
            
            # Calculate weighted average
            if type_data and sum(data['conc'] for data in melted_type_data) > 0:
                weighted_avg = sum(data['ratio'] * data['conc'] for data in type_data) / sum(data['conc'] for data in melted_type_data)
            else:
                weighted_avg = np.nan
            cycle_values[cycle][oligo_type['label']] = weighted_avg

    # Plot the values for each oligo type across cycles
    for oligo_type in oligo_types:
        label = oligo_type['label']
        values = [cycle_values[cycle].get(label, 0) for cycle in all_cycles]
        if oligo_type['is_clockwise'] is None:
            plt.plot(all_cycles, np.array(values), 
                    marker=oligo_type['marker'], 
                    color=oligo_type['color'], 
                    label=label, 
                    linestyle=oligo_type['linestyle'])
    
    plt.xlabel('PCR Cycle')
    plt.ylabel('Duplex Weighted Average Productive Ratio')
    plt.title('Duplex-Weighted Productive Ratio by Oligo Type Across Cycles')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(output_dir, f'{prefix}duplex_weighted_ratio_summary_by_oligo_type.png')
    pdf_path = os.path.join(output_dir, f'{prefix}duplex_weighted_ratio_summary_by_oligo_type.pdf')
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()


def plot_individual_avg_ratio(mutant_analysis, output_dir='.', prefix=''):
    """
    Create individual plots for mutant and wild-type oligos showing average ratio vs distance
    """
    # Get all available cycles and distances
    all_cycles = sorted(set([data['cycle'] for data in mutant_analysis]))
    all_distances = sorted(set([data['mutation_distance'] for data in mutant_analysis 
                               if data['mutation_distance'] is not None]))
    
    # Set up a colormap for cycles
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / len(all_cycles)) for i in range(len(all_cycles))]
    
    # Individual plots for each oligo type
    for oligo_type in ['mutant', 'wildtype']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot CW data
        for i, cycle in enumerate(all_cycles):
            # Get data for this cycle and oligo type
            cycle_data = [data for data in mutant_analysis 
                         if data['cycle'] == cycle and data['oligo_type'] == oligo_type]
            
            # Split CW data
            cw_cycle_data = [data for data in cycle_data 
                             if data['mutation_distance'] is not None and data['oligo'][2]]
            
            if cw_cycle_data:
                # Group by distance and calculate average ratio (simple average)
                avg_ratios = {}
                for distance in all_distances:
                    distance_data = [data for data in cw_cycle_data if data['mutation_distance'] == distance]
                    if distance_data:
                        avg_ratios[distance] = sum([data['ratio'] for data in distance_data]) / len(distance_data)
                
                if avg_ratios:
                    distances = sorted(avg_ratios.keys())
                    ratios = [avg_ratios[d] for d in distances]
                    ax1.plot(distances, ratios, marker='o', linestyle='-', color=colors[i], 
                            label=f'Cycle {cycle}')
        
        ax1.set_xlabel('Distance of Mutation from 3\' End')
        ax1.set_ylabel('Average Productive Ratio')
        ax1.set_title(f'Clockwise {oligo_type.capitalize()} Oligos - Average Ratio vs Distance')
        ax1.grid(alpha=0.3)
        if len(all_cycles) > 10:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax1.legend()
        
        # Plot CCW data
        for i, cycle in enumerate(all_cycles):
            # Get data for this cycle and oligo type
            cycle_data = [data for data in mutant_analysis 
                         if data['cycle'] == cycle and data['oligo_type'] == oligo_type]
            
            # Split CCW data
            ccw_cycle_data = [data for data in cycle_data 
                              if data['mutation_distance'] is not None and not data['oligo'][2]]
            
            if ccw_cycle_data:
                # Group by distance and calculate average ratio (simple average)
                avg_ratios = {}
                for distance in all_distances:
                    distance_data = [data for data in ccw_cycle_data if data['mutation_distance'] == distance]
                    if distance_data:
                        avg_ratios[distance] = sum([data['ratio'] for data in distance_data]) / len(distance_data)
                
                if avg_ratios:
                    distances = sorted(avg_ratios.keys())
                    ratios = [avg_ratios[d] for d in distances]
                    ax2.plot(distances, ratios, marker='o', linestyle='-', color=colors[i], 
                            label=f'Cycle {cycle}')
        
        ax2.set_xlabel('Distance of Mutation from 3\' End')
        ax2.set_ylabel('Average Productive Ratio')
        ax2.set_title(f'Counterclockwise {oligo_type.capitalize()} Oligos - Average Ratio vs Distance')
        ax2.grid(alpha=0.3)
        if len(all_cycles) > 10:
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax2.legend()
        
        plt.tight_layout()
        png_path = os.path.join(output_dir, f'{prefix}{oligo_type}_avg_ratio_vs_distance_per_cycle_cw_ccw.png')
        pdf_path = os.path.join(output_dir, f'{prefix}{oligo_type}_avg_ratio_vs_distance_per_cycle_cw_ccw.pdf')
        plt.savefig(png_path, dpi=300)
        plt.savefig(pdf_path)
        plt.close()


def plot_duplex_weighted_ratio_difference(mutant_analysis, melted_mutant_analysis, output_dir='.', prefix=''):
    """
    Create a plot showing the difference between mutant and wild-type weighted productive ratios across cycles
    """
    # Get all available cycles
    all_cycles = sorted(set([data['cycle'] for data in mutant_analysis]))
    
    plt.figure(figsize=(6, 5))
    
    # Calculate weighted averages for mutant and wild-type oligos in each cycle
    cycle_differences = []
    
    for cycle in all_cycles:
        cycle_data = [data for data in mutant_analysis if data['cycle'] == cycle]
        melted_cycle_data = [data for data in melted_mutant_analysis if data['cycle'] == cycle]
        
        # Get all mutant data for this cycle
        mutant_type_data = [data for data in cycle_data 
                          if data['oligo_type'] == 'mutant' and
                          data['mutation_distance'] is not None]
        
        mutant_melted_type_data = [data for data in melted_cycle_data 
                                 if data['oligo_type'] == 'mutant' and
                                 data['mutation_distance'] is not None]
        
        # Get all wild-type data for this cycle
        wildtype_type_data = [data for data in cycle_data 
                            if data['oligo_type'] == 'wildtype' and
                            data['mutation_distance'] is not None]
        
        wildtype_melted_type_data = [data for data in melted_cycle_data 
                                   if data['oligo_type'] == 'wildtype' and
                                   data['mutation_distance'] is not None]
        
        # Calculate weighted averages
        mutant_weighted_avg = np.nan
        if mutant_type_data and sum(data['conc'] for data in mutant_melted_type_data) > 0:
            mutant_weighted_avg = sum(data['ratio'] * data['conc'] for data in mutant_type_data) / sum(data['conc'] for data in mutant_melted_type_data)
        
        wildtype_weighted_avg = np.nan
        if wildtype_type_data and sum(data['conc'] for data in wildtype_melted_type_data) > 0:
            wildtype_weighted_avg = sum(data['ratio'] * data['conc'] for data in wildtype_type_data) / sum(data['conc'] for data in wildtype_melted_type_data)
        
        # Calculate difference (mutant - wild-type)
        if not np.isnan(mutant_weighted_avg) and not np.isnan(wildtype_weighted_avg):
            difference = mutant_weighted_avg - wildtype_weighted_avg
        else:
            difference = np.nan
        
        cycle_differences.append(difference)
    
    # Plot the difference curves
    plt.plot(all_cycles, np.log10(np.array(cycle_differences)), 
             marker='o', 
             color='#fbb040', 
             label='Mutant - Wild-Type Difference', 
             linestyle='-',
             linewidth=2)

    plt.plot(all_cycles, np.log10(-np.array(cycle_differences)), 
            marker='o', 
            color='#00a651', 
            label='Wild-Type - Mutant Difference', 
            linestyle='-',
            linewidth=2)
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.xlabel('PCR Cycle')
    plt.ylabel('Difference in Duplex Weighted Average Productive Ratio')
    plt.title('Mutant vs Wild-Type Productive Ratio Difference Across Cycles')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, f'{prefix}duplex_weighted_ratio_difference.png')
    pdf_path = os.path.join(output_dir, f'{prefix}duplex_weighted_ratio_difference.pdf')
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()


def plot_results(mutant_analysis, melted_mutant_analysis, output_dir='.', timestamp=None):
    """
    Create all plots for the analysis results
    """
    if not mutant_analysis:
        print("No data to plot")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the last cycle 
    last_cycle = max([data['cycle'] for data in mutant_analysis])
    if not any(data['cycle'] == last_cycle for data in mutant_analysis):
        print(f"No data for last cycle {last_cycle}")
        return
    
    # Create the timestamp prefix for filenames if provided
    prefix = f"vcg_data{timestamp}_Figure_" if timestamp else ""
    
    # Generate all plots
    print("  Generating mutant scatter plots...")
    plot_mutant_scatter(mutant_analysis, last_cycle, output_dir, prefix)
    
    print("  Generating wild-type scatter plots...")
    plot_wildtype_scatter(mutant_analysis, last_cycle, output_dir, prefix)
    
    print("  Generating combined average ratio plots...")
    plot_combined_avg_ratio(mutant_analysis, output_dir, prefix)
    
    print("  Generating weighted ratio summary plot...")
    plot_weighted_ratio_summary(mutant_analysis, output_dir, prefix)

    print("  Generating duplex-weighted ratio summary plot...")
    plot_duplex_weighted_ratio_summary(mutant_analysis, melted_mutant_analysis, output_dir, prefix)
    
    print("  Generating duplex-weighted ratio difference plot...")
    plot_duplex_weighted_ratio_difference(mutant_analysis, melted_mutant_analysis, output_dir, prefix)


def main():
    parser = argparse.ArgumentParser(description='Analyze PCR cycle data for mutant oligos')
    parser.add_argument('pickle_file', help='Path to the pickle file with simulation data')
    parser.add_argument('--output-dir', default='plots', help='Directory to save the plots')
    parser.add_argument('--min-overlap', type=int, default=2, help='Minimum overlap required for annealing (default: 3)')
    args = parser.parse_args()
    
    print(f"Loading data from {args.pickle_file}...")
    state_data = load_system_state(args.pickle_file)
    melted_data = melt_all_duplexes(state_data)
    
    # Extract timestamp from filename if possible
    timestamp = None
    match = re.search(r'vcg_data(_\d{4}_\d{4})\.pkl', args.pickle_file)
    if match:
        timestamp = match.group(1)
    
    print(f"Analyzing PCR cycles with min_overlap={args.min_overlap}...")
    mutant_analysis = analyze_pcr_cycles(state_data, min_overlap=args.min_overlap)
    melted_mutant_analysis = analyze_pcr_cycles(melted_data, min_overlap=args.min_overlap)
    
    print(f"Generating plots in {args.output_dir}...")
    plot_results(mutant_analysis, melted_mutant_analysis, args.output_dir, timestamp)
    
    print("Done!")

if __name__ == "__main__":
    main() 