# Use this file for the full Extended OligomerSystem class
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import pickle

from vcg_core_fns import CoreOligomerSystem

class ExtendedOligomerSystem(CoreOligomerSystem):
        
    def __init__(self, genome_length):
        super().__init__(genome_length)

    def generate_tiling_oligomers(self, oligo_length, shift_oligo, misaligned_CW_CCW, 
                            global_offset = 0,concentration=1.0, is_duplex=False, is_mutant=0, reactant1_idx=None, reactant2_idx=None):
        """Generate oligomers that tile a circular genome with parameters for add_oligomer_type."""
        oligomers = []
        
        # Generate clockwise oligomers
        for start_pos in range(0, self.genome_length, shift_oligo):
            start = (start_pos + global_offset) % self.genome_length
            end = (start + oligo_length - 1) % self.genome_length
            oligomers.append((start, end, True, concentration, is_duplex, is_mutant, reactant1_idx, reactant2_idx))
            
        # Generate counterclockwise oligomers - ensuring correct length for _region_length
        for start_pos in range(misaligned_CW_CCW, self.genome_length, shift_oligo):
            # Apply the same global offset as for clockwise oligomers
            start = (start_pos + global_offset) % self.genome_length
            # For CCW oligos with _region_length(start, end) = oligo_length, we need:
            # If end >= start: end - start + 1 = oligo_length  => end = start + oligo_length - 1
            # If end < start: (end + genome_length) - start + 1 = oligo_length => end = start - genome_length + oligo_length - 1
            # We want end < start for CCW, so we use the second formula
            end = (start - self.genome_length + oligo_length - 1) % self.genome_length
            
            # Verify this gives the right length with _region_length
            # length = (end + genome_length) - start + 1 if end < start
            # length = (end + genome_length) - start + 1 = (start - genome_length + oligo_length - 1 + genome_length) - start + 1 = oligo_length
            
            oligomers.append((start, end, False, concentration, is_duplex, is_mutant, reactant1_idx, reactant2_idx))
        
        return oligomers


    def print_system_state(self, visualization_threshold=0, max_oligos=float('inf'), multi_track=True, justtable=False, sort_by_concentration=False, visualize_genome_only=False):
        """
        Print the current state of the system with comprehensive visualization.
        Shows 5' and 3' ends correctly for CW and CCW strands.
        
        Args:
            visualization_threshold: Minimum concentration to display (default 0 to show all oligos)
            max_oligos: Maximum number of oligomers to display in the visualization
            multi_track: Whether to display the multi-track visualization
            justtable: If True, only print the oligomers table
            sort_by_concentration: If True, sort oligomers by concentration (descending) instead of by index
            visualize_genome_only: If True, only print the genome visualization without multi-track or table
        """
        #print("\n--- system state ---")
        
        # Define mutation site at the beginning
        mutation_site = self.genome_length // 2
        
        # Get oligomers with non-zero concentration that aren't DUPLEX_COMPONENT_MARKER (-1)
        visible_oligos = [(i, oligo_type, self.concentrations[i]) 
                        for i, oligo_type in enumerate(self.current_oligomers)
                        if i < len(self.concentrations) and 
                        self.concentrations[i] > visualization_threshold and 
                        self.concentrations[i] != self.DUPLEX_COMPONENT_MARKER]
        
        # Sort by concentration (descending) or index (ascending) based on parameter
        if sort_by_concentration:
            visible_oligos.sort(key=lambda x: x[2], reverse=True)
        else:
            visible_oligos.sort(key=lambda x: x[0])
        
        # Only limit by max_oligos parameter (no other restrictions)
        display_oligos = visible_oligos[:max_oligos] if len(visible_oligos) > max_oligos else visible_oligos
        
        # Calculate total concentration
        total_conc = np.sum([c for c in self.concentrations if c > 0 and c != self.DUPLEX_COMPONENT_MARKER])
        
        # If visualize_genome_only is True, skip statistics and just visualize the genome
        if visualize_genome_only:
            # Create a genome visualization
            self._visualize_genome(display_oligos)
            return
            
        # Separate by type for statistics - using first 5 elements of tuple for type determination
        ss_cw = [o for o in visible_oligos if not o[1][3] and o[1][2]]       # Single-stranded CW
        ss_ccw = [o for o in visible_oligos if not o[1][3] and not o[1][2]]  # Single-stranded CCW
        dup_cw = [o for o in visible_oligos if o[1][3] and o[1][2]]          # Duplex CW
        dup_ccw = [o for o in visible_oligos if o[1][3] and not o[1][2]]     # Duplex CCW
        mutant_oligos = [o for o in visible_oligos if o[1][4] > 0]
        
        # Print compact summary
        print(f"Found: {len(visible_oligos)} oligos (showing {len(display_oligos)}). " +
            f"Total conc: {total_conc:.2f}, Single: {len(ss_cw+ss_ccw)}, Duplex: {len(dup_cw+dup_ccw)}, Mutant: {len(mutant_oligos)}")
        
        # Create compact header for the table - added AnnLen column
        print(f"{'Idx':<5}{'Start':<6}{'End':<6}{'Dir':<5}{'Type':<8}{'Len':<5}{'AnnLen':<7}{'Mut':<5}{'R1':<5}{'R2':<5}{'Conc':<8}{'%':<5}")
        print("-" * 70)  # Extended the line to match the header width
        
        # Helper function to calculate annealing length for a duplex
        def get_oligo_annealing_length(oligo_type):
            # If not a duplex, return "-"
            if not oligo_type[3]:
                return "-"
                
            # Get reactant indices
            if len(oligo_type) >= 7:
                r1_idx, r2_idx = oligo_type[5], oligo_type[6]
                if (r1_idx is not None and r1_idx < len(self.current_oligomers) and
                    r2_idx is not None and r2_idx < len(self.current_oligomers)):
                    
                    # Get reactant oligomers
                    r1 = self.current_oligomers[r1_idx]
                    r2 = self.current_oligomers[r2_idx]
                    
                    # Use the centralized efficient method
                    return self.calculate_annealing_length(r1, r2)
            
            return "?"  # If we can't determine the annealing length
        
        # Display oligomers in table format
        for i, oligo_type, conc in display_oligos:
            # Extract basic properties (first 5 elements)
            start, end, is_clockwise, is_duplex, mutation_status = oligo_type[:5]
            
            # Extract reactant indices if available (elements 5 and 6)
            r1_idx = oligo_type[5] if len(oligo_type) > 5 else None
            r2_idx = oligo_type[6] if len(oligo_type) > 6 else None
            
            # Compact formatting
            direction = "CW" if is_clockwise else "CCW"
            status = "Dup" if is_duplex else "Single"
            
            # Short mutation text
            if is_duplex:
                mutation_text = {0: "-", 1: "S1", 2: "S2", 3: "Both"}.get(mutation_status, "?")
            else:
                mutation_text = "Yes" if mutation_status else "-"
            
            # Calculate length accounting for circularity
            length = end - start + 1 if end >= start else (self.genome_length - start) + (end + 1)
            
            # Calculate annealing length for duplexes
            annealing_length = get_oligo_annealing_length(oligo_type)
            
            # Percentage of total
            percent = (conc / total_conc * 100) if total_conc > 0 else 0
            
            # Reactant indices display - show "-" if None
            r1_display = str(r1_idx) if r1_idx is not None else "-"
            r2_display = str(r2_idx) if r2_idx is not None else "-"
            
            # Print with concentration formatted to 3 decimal places
            print(f"{i:<5}{start:<6}{end:<6}{direction:<5}{status:<8}{length:<5}{annealing_length:<7}{mutation_text:<5}{r1_display:<5}{r2_display:<5}{conc:<8.3f}{percent:<5.1f}%")
        
        if justtable:  # If justtable is requested, return here
            return
        
        # Visual genome representation
        self._visualize_genome(display_oligos)
        
        # Multi-track visualization if requested
        if multi_track:
            print("\n=== MULTI-TRACK VIEW ===")
            
            # Function to create track visualization - streamlined
            def create_track_viz(oligos, track_name):
                if not oligos:
                    return
                    
                print(f"{track_name}:")
                
                # Create multiple tracks as needed (no track limit)
                tracks = []
                max_tracks = 10  # Increased to support more visualization
                
                for idx, oligo_type, conc in oligos:
                    start, end = oligo_type[0], oligo_type[1]
                    is_clockwise = oligo_type[2]
                    
                    # Try to fit this oligomer into an existing track
                    placed = False
                    for track_idx, track in enumerate(tracks):
                        overlap = any((t_start <= end and start <= t_end) or 
                                    (t_end < t_start and (start <= t_end or t_start <= end)) or
                                    (end < start and (t_start <= end or start <= t_end))
                                    for t_start, t_end, _ in track)
                        
                        if not overlap:
                            track.append((start, end, is_clockwise))
                            placed = True
                            break
                    
                    # If couldn't place in existing tracks and not at max, create new track
                    if not placed and len(tracks) < max_tracks:
                        tracks.append([(start, end, is_clockwise)])
                        placed = True
                
                # Visualize each track
                for track_idx, track in enumerate(tracks):
                    track_viz = ['·'] * self.genome_length
                    track_viz[mutation_site] = 'X'  # Using mutation_site from the outer scope
                    
                    # Mark positions in track
                    for start, end, is_clockwise in track:
                        if end >= start:
                            for pos in range(start, end + 1):
                                if pos < self.genome_length:
                                    track_viz[pos] = 'O'
                        else:  # Wraps around
                            for pos in range(start, self.genome_length):
                                track_viz[pos] = 'O'
                            for pos in range(0, end + 1):
                                track_viz[pos] = 'O'
                        
                        # Add 5' and 3' markers based on direction
                        if is_clockwise:
                            # CW: 5' at start, 3' at end
                            if start < self.genome_length:
                                track_viz[start] = '5'
                            if end < self.genome_length:
                                track_viz[end] = '3'
                        else:
                            # CCW: 3' at start, 5' at end
                            if start < self.genome_length:
                                track_viz[start] = '3'
                            if end < self.genome_length:
                                track_viz[end] = '5'
                    
                    print(f"  Track {track_idx+1}: " + "".join(track_viz))
            
            # Further separate by mutation status
            ss_cw_mut = [o for o in ss_cw if o[1][4]]
            ss_cw_norm = [o for o in ss_cw if not o[1][4]]
            ss_ccw_mut = [o for o in ss_ccw if o[1][4]]
            ss_ccw_norm = [o for o in ss_ccw if not o[1][4]]
            
            # Create visualizations for each type - using abbreviated titles
            create_track_viz(ss_cw_norm, "Single CW (normal)")
            create_track_viz(ss_cw_mut, "Single CW (mutant)")
            create_track_viz(ss_ccw_norm, "Single CCW (normal)")
            create_track_viz(ss_ccw_mut, "Single CCW (mutant)")
            
            # For duplexes, group by mutation status with simplified names
            dup_groups = {
                "Normal": [d for d in dup_cw + dup_ccw if d[1][4] == 0],
                "Strand1 mut": [d for d in dup_cw + dup_ccw if d[1][4] == 1],
                "Strand2 mut": [d for d in dup_cw + dup_ccw if d[1][4] == 2],
                "Both strands mut": [d for d in dup_cw + dup_ccw if d[1][4] == 3]
            }
            
            # Visualize each duplex group
            for label, duplexes in dup_groups.items():
                if duplexes:
                    create_track_viz(duplexes, f"Duplex ({label})")

    def _capture_current_state(self, label):
        """
        Captures the current state of the PCR system for visualization.
        
        Args:
            label: A label for this state (e.g., "initial", "cycle_1")
        """
        # Create state record with complete oligomer and concentration data
        state = {
            'label': label,
            'oligomers': self.current_oligomers.copy(),  # Store complete oligomer list
            'concentrations': self.concentrations.copy() if self.concentrations is not None else np.array([])  # Store complete concentration array
        }
        
        # Store the state
        if not hasattr(self, 'cycle_states'):
            self.cycle_states = []
        
        self.cycle_states.append(state)



    def visualize_pcr_results(self, save_plots=False, show_plots=False, output_dir='plots'):
        """
        Visualize the results of PCR simulation.
        
        Args:
            save_plots: Whether to save the plots to files
            show_plots: Whether to display the plots interactively
            output_dir: Directory to save output plots (default: 'plots')
        """
        # Check if we have cycle states to visualize
        if not hasattr(self, 'cycle_states'):
            return
        if not self.cycle_states:
            return
        
        # Extract data
        cycles = []
        total_concs = []
        ss_concs = []
        duplex_concs = []
        total_mutant_concs = []
        ss_mutant_concs = []
        duplex_one_mut_concs = []
        duplex_both_mut_concs = []
        duplex_no_mut_concs = []  # Track duplexes with no mutations
        
        # Data for new plot - oligomer types with concentration > 0.1% of total
        oligo_type_counts = []
        ss_normal_counts = []
        ss_mutant_counts = []
        duplex_normal_counts = []
        duplex_one_mut_counts = []
        duplex_both_mut_counts = []
        
        # Process each saved state
        for i, state_data in enumerate(self.cycle_states):
            # Extract label and state from the saved data
            if isinstance(state_data, dict):
                label = state_data.get('label', f"State {i}")
                state = state_data.get('concentrations', [])
                oligomers = state_data.get('oligomers', [])
            else:
                continue
            
            # Process both initial state and cycle states
            if label == "initial":
                cycle_num = 0
                cycles.append(cycle_num)
            elif label.startswith("cycle_"):  # Match lowercase "cycle_" prefix
                try:
                    cycle_num = int(label.split('_')[1])
                    cycles.append(cycle_num)
                except (IndexError, ValueError) as e:
                    cycles.append(i)  # Fallback to index if can't parse cycle number
            else:
                continue
            
            # Initialize counters for this cycle
            total_conc = 0
            ss_conc = 0
            duplex_conc = 0
            total_mutant_conc = 0
            ss_mutant_conc = 0
            duplex_one_mut = 0
            duplex_both_mut = 0
            duplex_no_mut = 0  # Duplexes with no mutations
            
            # Counters for oligomer types with conc > 0.1% of total
            ss_normal_count = 0
            ss_mutant_count = 0
            duplex_normal_count = 0
            duplex_one_mut_count = 0
            duplex_both_mut_count = 0
            
            # Process each oligomer and its concentration
            for idx, c in enumerate(state):
                if idx >= len(oligomers) or c <= 0 or c == self.DUPLEX_COMPONENT_MARKER:
                    continue
                
                oligo_type = oligomers[idx]
                total_conc += c
                
                # Check if it's a duplex (index 3 is is_duplex)
                if oligo_type[3]:
                    duplex_conc += c
                    
                    # Check mutation status for duplex (index 4 is is_mutant)
                    if oligo_type[4] == 1:
                        duplex_one_mut += c
                        total_mutant_conc += c
                    elif oligo_type[4] == 2:
                        duplex_one_mut += c
                        total_mutant_conc += c
                    elif oligo_type[4] == 3:
                        duplex_both_mut += c
                        total_mutant_conc += c
                    else:
                        duplex_no_mut += c  # Count duplexes with no mutations
                else:
                    ss_conc += c
                    if oligo_type[4] > 0:
                        ss_mutant_conc += c
                        total_mutant_conc += c
            
            # Calculate which oligomers have concentration > 0.1% of total
            threshold_conc = total_conc * 0.001  # 0.1% threshold
            
            for idx, c in enumerate(state):
                if idx >= len(oligomers) or c <= threshold_conc:
                    continue
                
                oligo_type = oligomers[idx]
                
                # Categorize by type
                if oligo_type[3]:  # Duplex
                    if oligo_type[4] == 0:
                        duplex_normal_count += 1
                    elif oligo_type[4] == 1 or oligo_type[4] == 2:
                        duplex_one_mut_count += 1
                    elif oligo_type[4] == 3:
                        duplex_both_mut_count += 1
                else:  # Single-stranded
                    if oligo_type[4] == 0:
                        ss_normal_count += 1
                    else:
                        ss_mutant_count += 1
            
            total_concs.append(total_conc)
            ss_concs.append(ss_conc)
            duplex_concs.append(duplex_conc)
            total_mutant_concs.append(total_mutant_conc)
            ss_mutant_concs.append(ss_mutant_conc)
            duplex_one_mut_concs.append(duplex_one_mut)
            duplex_both_mut_concs.append(duplex_both_mut)
            duplex_no_mut_concs.append(duplex_no_mut)
            
            # Store counts for new plot
            oligo_type_counts.append(ss_normal_count + ss_mutant_count + duplex_normal_count + 
                                    duplex_one_mut_count + duplex_both_mut_count)
            ss_normal_counts.append(ss_normal_count)
            ss_mutant_counts.append(ss_mutant_count)
            duplex_normal_counts.append(duplex_normal_count)
            duplex_one_mut_counts.append(duplex_one_mut_count)
            duplex_both_mut_counts.append(duplex_both_mut_count)
        
        # Skip visualization if no cycle data was found
        if not cycles:
            return
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Mutation Analysis (left panel)
        # Calculate "normal" non-mutant single stranded concentration
        normal_ss_concs = [ss - mut_ss for ss, mut_ss in zip(ss_concs, ss_mutant_concs)]
        
        ax1.plot(cycles, total_concs, color='#7D7D7D', linewidth=2, label='Total')  # Dark Gray
        ax1.plot(cycles, normal_ss_concs, color='#0072B2', linewidth=2, label='Normal SS')  # Blue
        ax1.plot(cycles, total_mutant_concs, color='#D9D9D9', linewidth=2, label='Total Mutant')  # Light Gray
        ax1.plot(cycles, ss_mutant_concs, color='#E69F00', label='SS with Mutation')  # Orange
        ax1.plot(cycles, duplex_one_mut_concs, color='#009E73', label='Duplex (One Strand Mutant)')  # Green
        ax1.plot(cycles, duplex_both_mut_concs, color='#CC79A7', label='Duplex (Both Strands Mutant)')  # Purple
        ax1.plot(cycles, duplex_no_mut_concs, color='#56B4E9', label='Duplex (No Mutations)')  # Cyan
        ax1.set_xlabel('PCR Cycle')
        ax1.set_ylabel('Concentration')
        ax1.set_title('Mutation Analysis')
        ax1.set_yscale('log')  # Set y-axis to logarithmic scale
        
        # Move the legend outside the plot area
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize='small')
        ax1.grid(True)
        
        # Plot 2: Oligomer Type Counts (right panel)
        width = 0.15  # the width of the bars
        x = np.arange(len(cycles))  # the label locations
        
        ax2.bar(x, ss_normal_counts, width, label='SS (No Mutation)', color='#0072B2')  # Blue
        ax2.bar(x, ss_mutant_counts, width, bottom=ss_normal_counts, label='SS (With Mutation)', color='#E69F00')  # Orange
        
        # Calculate cumulative sums for stacking
        ss_total = [a + b for a, b in zip(ss_normal_counts, ss_mutant_counts)]
        duplex_one_bottom = ss_total.copy()
        duplex_both_bottom = [a + b for a, b in zip(duplex_one_bottom, duplex_one_mut_counts)]
        
        ax2.bar(x, duplex_normal_counts, width, bottom=ss_total, label='Duplex (No Mutation)', color='#56B4E9')  # Cyan
        ax2.bar(x, duplex_one_mut_counts, width, bottom=[a + b for a, b in zip(ss_total, duplex_normal_counts)], 
                label='Duplex (One Reactant Mutated)', color='#009E73')  # Green
        ax2.bar(x, duplex_both_mut_counts, width, bottom=duplex_both_bottom, 
                label='Duplex (Both Reactants Mutated)', color='#CC79A7')  # Purple
        
        ax2.set_xlabel('PCR Cycle')
        ax2.set_ylabel('Count of Oligomer Types (>0.1% conc.)')
        ax2.set_title('Oligomer Types by Category')
        ax2.set_xticks(x)
        ax2.set_xticklabels(cycles)
        
        # Move the legend outside the plot area
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize='small')
        ax2.grid(True, axis='y')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots:
            try:
                import os
                os.makedirs(output_dir, exist_ok=True)
                fig.savefig(os.path.join(output_dir, 'pcr_results.png'), dpi=150)
            except Exception:
                pass
        
        # Show plot if requested
        if show_plots:
            plt.show()
            
        # Close the figure
        plt.close(fig)


    def nucleate_de_novo_oligos(self, oligo_length=6, cleanup_threshold=0, verbose=False, seed=None, 
                                K=1.0, Pmax=1.0, sparse_sampling=0.05):
        """
        Generate de novo oligos that can spontaneously nucleate on exposed single-stranded regions.
        Uses a concentration-dependent probabilistic model for nucleation with saturation effects.
        
        Args:
            oligo_length: Length of the de novo oligos to create (default: 6)
            cleanup_threshold: Minimum concentration of new oligos
            verbose: Whether to print detailed output
            seed: Random seed for reproducibility
            K: Concentration parameter in saturation model (default: 1.0)
            Pmax: Maximum probability of nucleation at a site (default: 1.0)
            sparse_sampling: Fraction of potential nucleation sites to actually nucleate (default: 0.05 or 5%)
            
        Returns:
            Dictionary mapping new oligo types to their concentrations
        """
        # Set random seed for reproducibility if provided
        rng = np.random.RandomState(seed)
        
        if verbose:
            print(f"\nSimulating de novo nucleation of {oligo_length}-mers on single-stranded regions...")
            print(f"Using saturation model: p_c = Pmax*K/(K + C_total)")
            print(f"  where Pmax = {Pmax:.4f} (maximum nucleation probability)")
            print(f"  and K = {K:.4f} (concentration parameter)")
            print(f"Will sample {sparse_sampling:.1%} of potential nucleation sites")
            print(f"Minimum concentration threshold for new oligos: {cleanup_threshold}")
        
        # Track newly created oligos
        new_oligos = {}
        
        # Helper function to get positions covered by an oligo
        def get_positions(start, end):
            if end >= start:
                return set(range(start, end + 1))
            return set(range(start, self.genome_length)) | set(range(0, end + 1))
        
        # Define mutation site
        mutation_site = self.genome_length // 2
        
        # 1. First pass: identify all potential nucleation sites and their concentrations
        nucleation_sites = []  # List of (start, end, direction, is_mutant, concentration, template_type, template_idx)
        
        # Process single-stranded oligos
        single_stranded = self.get_single_stranded_oligos(0) # get all single stranded oligos, no threshold
        for oligo_idx, (oligo_tuple, conc) in enumerate(single_stranded):
            start, end, is_clockwise, _, is_mutant = oligo_tuple[:5]
            length = self._region_length(start, end)
            
            # Skip if too short for nucleation
            if length < oligo_length:
                continue
                
            # Get all positions covered by this oligo
            positions = get_positions(start, end)
            
            # For each possible starting position
            for pos in range(self.genome_length):
                # Check if this position and the next oligo_length-1 positions are in the template
                nucleation_region = set(range(pos, pos + oligo_length))
                if pos + oligo_length > self.genome_length:
                    nucleation_region = set(range(pos, self.genome_length)) | set(range(0, (pos + oligo_length) % self.genome_length))
                
                # Only consider creating oligo if all positions are in the template
                if nucleation_region.issubset(positions):
                    # Create a new oligo with the opposite direction of the template
                    new_oligo_direction = not is_clockwise
                    
                    # Check if this oligo covers the mutation site AND the template is a mutant
                    new_oligo_is_mutant = 1 if (mutation_site in nucleation_region and is_mutant == 1) else 0
                    
                    # Add this nucleation site to our list
                    nucleation_sites.append((
                        pos,  # start
                        (pos + oligo_length - 1) % self.genome_length,  # end
                        new_oligo_direction,  # direction
                        new_oligo_is_mutant,  # is_mutant
                        conc,  # concentration of the template at this site
                        'ss',  # template type (single-stranded)
                        oligo_idx  # template index
                    ))
        
        # Process exposed regions in duplexes
        duplexes = self.get_duplexes(0) # get all duplexes, no threshold
        for duplex_idx, duplex_tuple, duplex_conc in duplexes:
            # Get reactant indices
            if len(duplex_tuple) >= 7:
                reactant1_idx, reactant2_idx = duplex_tuple[5], duplex_tuple[6]
            else:
                continue
            
            # Get the reactant tuples
            reactant1 = self.current_oligomers[reactant1_idx]
            reactant2 = self.current_oligomers[reactant2_idx]
            
            r1_start, r1_end, r1_direction, _, r1_mutation = reactant1[:5]
            r2_start, r2_end, r2_direction, _, r2_mutation = reactant2[:5]
            
            # Get positions covered by each reactant
            positions1 = get_positions(r1_start, r1_end)
            positions2 = get_positions(r2_start, r2_end)
            
            # Find exposed regions (in either reactant but not in both)
            exposed_positions1 = positions1 - positions2
            exposed_positions2 = positions2 - positions1
            
            # Process exposed regions in reactant 1
            if exposed_positions1:
                # Find contiguous regions
                regions = self._find_contiguous_regions(exposed_positions1)
                
                for region_start, region_end in regions:
                    # Only process regions long enough for nucleation
                    region_length = self._region_length(region_start, region_end)
                    if region_length >= oligo_length:
                        # Use the duplex concentration directly without scaling
                        region_conc = duplex_conc
                        
                        # Generate potential nucleation sites along this exposed region
                        for i in range(region_length - oligo_length + 1):
                            # Calculate start and end positions
                            oligo_start = (region_start + i) % self.genome_length
                            oligo_end = (oligo_start + oligo_length - 1) % self.genome_length
                            
                            # Get positions covered by this potential oligo
                            oligo_positions = get_positions(oligo_start, oligo_end)
                            
                            # Check if this oligo covers the mutation site AND the template is a mutant
                            oligo_is_mutant = 1 if (mutation_site in oligo_positions and r1_mutation == 1) else 0
                            
                            # Add this nucleation site to our list
                            nucleation_sites.append((
                                oligo_start,  # start
                                oligo_end,  # end
                                not r1_direction,  # direction (opposite of template)
                                oligo_is_mutant,  # is_mutant
                                region_conc,  # concentration (now just duplex_conc)
                                'template',  # template type (simplified from 'duplex1')
                                duplex_idx  # template index
                            ))
            
            # Process exposed regions in reactant 2
            if exposed_positions2:
                # Find contiguous regions
                regions = self._find_contiguous_regions(exposed_positions2)
                
                for region_start, region_end in regions:
                    # Only process regions long enough for nucleation
                    region_length = self._region_length(region_start, region_end)
                    if region_length >= oligo_length:
                        # Use the duplex concentration directly without scaling
                        region_conc = duplex_conc
                        
                        # Generate potential nucleation sites along this exposed region
                        for i in range(region_length - oligo_length + 1):
                            # Calculate start and end positions
                            oligo_start = (region_start + i) % self.genome_length
                            oligo_end = (oligo_start + oligo_length - 1) % self.genome_length
                            
                            # Get positions covered by this potential oligo
                            oligo_positions = get_positions(oligo_start, oligo_end)
                            
                            # Check if this oligo covers the mutation site AND the template is a mutant
                            oligo_is_mutant = 1 if (mutation_site in oligo_positions and r2_mutation == 1) else 0
                            
                            # Add this nucleation site to our list
                            nucleation_sites.append((
                                oligo_start,  # start
                                oligo_end,  # end
                                not r2_direction,  # direction (opposite of template)
                                oligo_is_mutant,  # is_mutant
                                region_conc,  # concentration (now just duplex_conc)
                                'template',  # template type (simplified from 'duplex2')
                                duplex_idx  # template index
                            ))
        
        # If no nucleation sites found, return empty dictionary
        if not nucleation_sites:
            if verbose:
                print("No potential nucleation sites found, skipping nucleation")
            return new_oligos
        
        # 2. Calculate total concentration of nucleation sites
        total_sites = len(nucleation_sites)
        C_total = sum(site[4] for site in nucleation_sites)
        
        if verbose:
            print(f"Found {total_sites} potential nucleation sites with total concentration C_total = {C_total:.6f}")
        
        # 3. Calculate probability of nucleation per site using updated saturation model
        # p_c = Pmax*K/(K + C_total)
        p_c = Pmax * K / (K + C_total)
        
        if verbose:
            print(f"Calculated nucleation probability per unit concentration: p_c = {p_c:.6f}")
        
        # 4. Calculate concentration of de novo oligos for each site
        valid_nucleation_sites = []
        for i in range(len(nucleation_sites)):
            site = nucleation_sites[i]
            C_i = site[4]  # concentration of nucleation site i
            # New oligo concentration = p_c * C_i
            new_conc = p_c * C_i
            
            # Only keep sites where the predicted concentration is above threshold
            if new_conc >= cleanup_threshold:
                valid_nucleation_sites.append(site + (new_conc,))
        
        if verbose:
            skipped_sites = len(nucleation_sites) - len(valid_nucleation_sites)
            if skipped_sites > 0:
                print(f"Skipped {skipped_sites} nucleation sites with predicted concentration below threshold {cleanup_threshold}")
            print(f"Proceeding with {len(valid_nucleation_sites)} valid nucleation sites")
        
        # If no valid sites after concentration threshold, return empty dictionary
        if not valid_nucleation_sites:
            if verbose:
                print("No nucleation sites with concentration above threshold, skipping nucleation")
            return new_oligos
            
        # 5. Calculate number of sites to sample
        num_sites_to_sample = min(max(1, int(sparse_sampling * total_sites)), len(valid_nucleation_sites))
        
        # 6. Sample sites based on their concentrations without replacement
        if verbose:
            print(f"Sampling {num_sites_to_sample} sites ({sparse_sampling:.1%} of {total_sites} total sites)")
        
        # Get the concentrations for sampling
        site_concentrations = np.array([site[7] for site in valid_nucleation_sites])
        
        # Check if any sites have non-zero concentration
        if np.sum(site_concentrations) > 0:
            # Normalize to probabilities
            probs = site_concentrations / np.sum(site_concentrations)
            
            # Sample indices without replacement according to concentrations
            sampled_indices = rng.choice(
                len(valid_nucleation_sites), 
                size=min(num_sites_to_sample, len(valid_nucleation_sites)), 
                replace=False, 
                p=probs
            )
            sampled_sites = [valid_nucleation_sites[i] for i in sampled_indices]
        else:
            # If all concentrations are zero, sample uniformly
            sampled_indices = rng.choice(
                len(valid_nucleation_sites),
                size=min(num_sites_to_sample, len(valid_nucleation_sites)),
                replace=False
            )
            sampled_sites = [valid_nucleation_sites[i] for i in sampled_indices]
            
        if verbose:
            # Count mutation types in sampled sites
            mutant_sites = sum(1 for site in sampled_sites if site[3] == 1)
            print(f"Of these, {mutant_sites} sites ({mutant_sites/num_sites_to_sample:.1%} if any) will create mutant oligos")
        
        # 7. Create the new oligos at the sampled sites
        total_new_conc = 0
        mutant_new_conc = 0
        for site in sampled_sites:
            start, end, direction, is_mutant, _, _, _, new_conc = site
            total_new_conc += new_conc
            if is_mutant:
                mutant_new_conc += new_conc
            
            # Add the new oligo to the system
            new_oligo_idx = self.add_oligomer_type(
                start, end, direction,
                new_conc, False, is_mutant
            )
            
            if verbose:
                mutant_status = "mutant" if is_mutant else "wildtype"
                covers_mutation_site = "covers L/2" if mutation_site in get_positions(start, end) else "doesn't cover L/2"
                print(f"  Created de novo oligo at index {new_oligo_idx}: ({start}, {end}, {direction}, {is_mutant}) with conc {new_conc:.6f} ({mutant_status}, {covers_mutation_site})")
            
            # Add to our tracking dictionary
            new_oligos[self.current_oligomers[new_oligo_idx]] = new_conc
        
        # Use rebuild=False when updating the reaction network to include the new oligos
        if new_oligos:
            if verbose:
                print(f"Created {len(new_oligos)} new de novo oligos with total concentration {total_new_conc:.6f}")
                if total_new_conc > 0:
                    print(f"Of these, {mutant_new_conc/total_new_conc:.1%} of the concentration is from mutant oligos")
                print(f"Nucleation summary: {total_sites} potential sites → {len(valid_nucleation_sites)} valid sites → {num_sites_to_sample} nucleated sites")
                print(f"Updating reaction network...")
            self.build_reaction_network(rebuild=False, verbose=verbose)
        else:
            if verbose:
                print("No new de novo oligos created")
        
        return new_oligos

    def compute_oligo_length_histograms(self, visualization_threshold=0, print_results=False):
        """
        Compute histograms of oligomer lengths with various filters.
        
        Args:
            visualization_threshold: Minimum concentration to consider (default 0)
            print_results: Whether to print the histogram results (default False)
            
        Returns:
            Dictionary with the following keys:
            - 'ss_count': List of (length, count) tuples for all single-stranded oligos
            - 'ss_conc': List of (length, total_concentration) tuples for all single-stranded oligos
            - 'ss_normal_count': List of (length, count) tuples for non-mutant single-stranded oligos
            - 'ss_normal_conc': List of (length, total_concentration) tuples for non-mutant single-stranded oligos
            - 'ss_mutant_count': List of (length, count) tuples for mutant single-stranded oligos
            - 'ss_mutant_conc': List of (length, total_concentration) tuples for mutant single-stranded oligos
            - 'all_mutant_count': List of (length, count) tuples for all mutant oligos (ss and in duplexes)
            - 'all_mutant_conc': List of (length, total_concentration) tuples for all mutant oligos (ss and in duplexes)
        """
        # Get all single-stranded oligos
        ss_oligos = self.get_single_stranded_oligos(visualization_threshold) # get all single stranded oligos
        
        # Create dictionaries to store counts and concentrations by length
        ss_count = defaultdict(int)
        ss_conc = defaultdict(float)
        ss_normal_count = defaultdict(int)
        ss_normal_conc = defaultdict(float)
        ss_mutant_count = defaultdict(int)
        ss_mutant_conc = defaultdict(float)
        all_mutant_count = defaultdict(int)
        all_mutant_conc = defaultdict(float)
        
        # Process single-stranded oligos
        for oligo_type, conc in ss_oligos:
            start, end, is_clockwise, _, is_mutant = oligo_type[:5]
            length = self._region_length(start, end)
            
            # Add to general ss counts and concentrations
            ss_count[length] += 1
            ss_conc[length] += conc
            
            # Add to mutant or non-mutant categories
            if is_mutant:
                ss_mutant_count[length] += 1
                ss_mutant_conc[length] += conc
                all_mutant_count[length] += 1
                all_mutant_conc[length] += conc
            else:
                ss_normal_count[length] += 1
                ss_normal_conc[length] += conc
        
        # Process duplexes to find mutant oligos in duplexes
        duplexes = self.get_duplexes(visualization_threshold)
        # Track processed reactants to avoid double-counting
        processed_reactants = set()
        
        for idx, duplex_type, duplex_conc in duplexes:
            # Only interested in duplexes that contain a mutant oligo
            if duplex_type[4] > 0:  # Has at least one mutant strand
                # Get reactant indices
                if len(duplex_type) >= 7:
                    reactant1_idx, reactant2_idx = duplex_type[5], duplex_type[6]
                    if (reactant1_idx is not None and reactant1_idx < len(self.current_oligomers) and
                        reactant2_idx is not None and reactant2_idx < len(self.current_oligomers)):
                        
                        # Get reactant oligomers
                        r1 = self.current_oligomers[reactant1_idx]
                        r2 = self.current_oligomers[reactant2_idx]
                        
                        # Only count the lengths of the mutant reactants
                        # Use reactant index to avoid double-counting
                        if r1[4] > 0 and (reactant1_idx, idx) not in processed_reactants:  # r1 is a mutant
                            length = self._region_length(r1[0], r1[1])
                            all_mutant_count[length] += 1
                            all_mutant_conc[length] += duplex_conc  # Use duplex concentration
                            processed_reactants.add((reactant1_idx, idx))
                            
                        if r2[4] > 0 and (reactant2_idx, idx) not in processed_reactants:  # r2 is a mutant
                            length = self._region_length(r2[0], r2[1])
                            all_mutant_count[length] += 1
                            all_mutant_conc[length] += duplex_conc  # Use duplex concentration
                            processed_reactants.add((reactant2_idx, idx))
        
        # Convert dictionaries to sorted lists of tuples
        result = {
            'ss_count': sorted(ss_count.items()),
            'ss_conc': sorted(ss_conc.items()),
            'ss_normal_count': sorted(ss_normal_count.items()),
            'ss_normal_conc': sorted(ss_normal_conc.items()),
            'ss_mutant_count': sorted(ss_mutant_count.items()),
            'ss_mutant_conc': sorted(ss_mutant_conc.items()),
            'all_mutant_count': sorted(all_mutant_count.items()),
            'all_mutant_conc': sorted(all_mutant_conc.items())
        }
        
        # Print results if requested
        if print_results:
            print("\n=== Oligo Length Distribution ===")
            
            # Print single-stranded oligos
            ss_data = [f"{length}: {count}, {conc:.2f} uM" for (length, count), (_, conc) in 
                      zip(result['ss_count'], result['ss_conc'])]
            print(f"All SS Oligos   (Length: Count, Conc): {' | '.join(ss_data)}")
            
            # Print normal vs mutant with concentrations - split into two lines
            normal_data = [f"{length}: {count}, {conc:.2f} uM" for (length, count), (_, conc) in 
                          zip(result['ss_normal_count'], result['ss_normal_conc'])]
            print(f"SS Normal Oligos(Length: Count, Conc): {' | '.join(normal_data)}")
            
            mutant_data = [f"{length}: {count}, {conc:.2f} uM" for (length, count), (_, conc) in 
                          zip(result['ss_mutant_count'], result['ss_mutant_conc'])]
            print(f"SS Mutant Oligos(Length: Count, Conc): {' | '.join(mutant_data)}")
            
            # Print all mutant oligos
            all_mut = [f"{length}: {count}, {conc:.2f} uM" for (length, count), (_, conc) in 
                      zip(result['all_mutant_count'], result['all_mutant_conc'])]
            print(f"Muts in ss + dup(Length: Count, Conc): {' | '.join(all_mut)}")
        
        return result
    
    def compute_allele_concentrations(self,  print_results=False):
        """
        Compute the total concentrations of mutant and wildtype alleles at the mutation site.
        
        Args:
            print_results: Whether to print the concentration results (default False)
            
        Returns:
            Dictionary with the following keys:
            - 'mutant_concentration': Total concentration of mutant alleles
            - 'wildtype_concentration': Total concentration of wildtype alleles at the mutation site
            - 'mutant_to_wildtype_ratio': Ratio of mutant to wildtype concentration
        """
        # Define mutation site
        mutation_site = self.genome_length // 2
        
        # Initialize concentration counters
        mutant_conc = 0.0
        wildtype_conc = 0.0
        
        # Helper function to check if a position is covered by an oligo
        def covers_position(start, end, position):
            if end >= start:
                return start <= position <= end
            else:  # Wraps around
                return start <= position or position <= end
        
        # Process single-stranded oligos
        ss_oligos = self.get_single_stranded_oligos(0)
        for oligo_type, conc in ss_oligos:
            start, end, _, _, is_mutant = oligo_type[:5]
            
            # Check if this oligo covers the mutation site
            if covers_position(start, end, mutation_site):
                if is_mutant:
                    mutant_conc += conc
                else:
                    wildtype_conc += conc
        
        # Process duplexes
        duplexes = self.get_duplexes(0)
        for _, duplex_type, duplex_conc in duplexes:
            # Get mutation status (0=none, 1=strand1, 2=strand2, 3=both)
            mutation_status = duplex_type[4]
            
            # Get reactant indices
            if len(duplex_type) >= 7:
                reactant1_idx, reactant2_idx = duplex_type[5], duplex_type[6]
                if (reactant1_idx is not None and reactant1_idx < len(self.current_oligomers) and
                    reactant2_idx is not None and reactant2_idx < len(self.current_oligomers)):
                    
                    # Get reactant oligomers
                    r1 = self.current_oligomers[reactant1_idx]
                    r2 = self.current_oligomers[reactant2_idx]
                    
                    # Check if reactants cover the mutation site
                    r1_covers = covers_position(r1[0], r1[1], mutation_site)
                    r2_covers = covers_position(r2[0], r2[1], mutation_site)
                    
                    # Add to appropriate counters based on mutation status
                    if mutation_status == 0:  # No mutations
                        if r1_covers:
                            wildtype_conc += duplex_conc
                        if r2_covers:
                            wildtype_conc += duplex_conc
                    elif mutation_status == 1:  # Strand 1 mutated
                        if r1_covers:
                            mutant_conc += duplex_conc
                        if r2_covers:
                            wildtype_conc += duplex_conc
                    elif mutation_status == 2:  # Strand 2 mutated
                        if r1_covers:
                            wildtype_conc += duplex_conc
                        if r2_covers:
                            mutant_conc += duplex_conc
                    elif mutation_status == 3:  # Both strands mutated
                        if r1_covers:
                            mutant_conc += duplex_conc
                        if r2_covers:
                            mutant_conc += duplex_conc
        
        # Calculate ratio (avoid division by zero)
        ratio = mutant_conc / wildtype_conc if wildtype_conc > 0 else float('inf')
        
        result = {
            'mutant_concentration': mutant_conc,
            'wildtype_concentration': wildtype_conc,
            'mutant_to_wildtype_ratio': ratio
        }
        
        # Print results if requested
        if print_results:
            print(f"Allele conc: Mutant = {mutant_conc:.4f} | " +
                  f"Wildtype = {wildtype_conc:.4f} | " +
                  f"Ratio (M/W) = {ratio:.4f}")
        
        return result

    def save_system_state(self, filename):
        """
        Save the current state of the system to a pickle file.
        
        Args:
            filename: Path where the pickle file will be saved
        """
        # Create a dictionary with all data to save
        state_data = {
            # System parameters
            'genome_length': self.genome_length,

            # Current state
            'oligomers': self.current_oligomers.copy(),
            'concentrations': self.concentrations.copy(),
            
            # Captured states if available
            'cycle_states': self.cycle_states if hasattr(self, 'cycle_states') else []
        }
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(state_data, f)
        
        print(f"System state saved to {filename}")

    @staticmethod
    def load_system_state(filename):
        """
        Load a saved system state from a pickle file.
        
        Args:
            filename: Path to the pickle file
            
        Returns:
            Dictionary containing the saved system state
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def _visualize_genome(self, display_oligos):
        """
        Display a visualization of the genome with oligomers represented.
        Shows 5' and 3' ends correctly for CW and CCW strands.
        For genomes longer than 60bp, each character represents 2 nucleotides.
        
        Args:
            display_oligos: List of tuples (idx, oligo_type, conc) to display
        """
        # Visual genome representation
        mutation_site = self.genome_length // 2
        PREFIX_WIDTH = 15  # Width for prefix text
        total_conc = np.sum([c for c in self.concentrations if c > 0 and c != self.DUPLEX_COMPONENT_MARKER])
        
        # Determine if we need to compress the visualization (2 nucleotides per character)
        compress_viz = self.genome_length > 60
        display_length = self.genome_length // 2 if compress_viz else self.genome_length
        
        print("\nGenome Visualization      Legend: 5/3=5'/3' ends O=Covered X=Mutation M=MutationCovered D>=DuplexCW S>=SingleCW")
        if compress_viz:
            print(f"Note: Each character represents 2 nucleotides (genome length: {self.genome_length}bp)")
        
        # Print position markers and ruler in one line
        markers = " " * PREFIX_WIDTH
        step = 20 if compress_viz else 10
        for i in range(0, self.genome_length+step, step):
            idx_str = str(i)
            markers += idx_str + " " * ((step // 2 if compress_viz else step) - len(idx_str)) if i + step <= self.genome_length else idx_str
        print(markers)
        
        # Create reference line with mutation site marked
        ref_line = ['·'] * display_length
        mutation_display_pos = mutation_site // 2 if compress_viz else mutation_site
        ref_line[mutation_display_pos] = 'X'
        print(f"{'Reference':>{PREFIX_WIDTH}} " + "".join(ref_line) + "  [Conc        |      % |Length      |Mutant   |Start-End]")
        
        # Helper function to calculate annealing length for a duplex
        def get_oligo_annealing_length(oligo_type):
            # If not a duplex, return "-"
            if not oligo_type[3]:
                return "-"
                
            # Get reactant indices
            if len(oligo_type) >= 7:
                r1_idx, r2_idx = oligo_type[5], oligo_type[6]
                if (r1_idx is not None and r1_idx < len(self.current_oligomers) and
                    r2_idx is not None and r2_idx < len(self.current_oligomers)):
                    
                    # Get reactant oligomers
                    r1 = self.current_oligomers[r1_idx]
                    r2 = self.current_oligomers[r2_idx]
                    
                    # Use the centralized efficient method
                    return self.calculate_annealing_length(r1, r2)
            
            return "?"  # If we can't determine the annealing length
        
        # Show each oligomer with fixed-width prefix
        for i, (idx, oligo_type, conc) in enumerate(display_oligos):
            # Extract basic properties (first 5 elements)
            start, end, is_clockwise, is_duplex, mutation_status = oligo_type[:5]
            
            # Create visualization of oligomer position
            genome_viz = ['·'] * display_length
            
            # Fill positions for this oligomer
            if compress_viz:
                # When compressed, each character represents 2 nucleotides
                if end >= start:
                    for pos in range(start, end + 1):
                        genome_viz[pos // 2] = 'O'
                else:  # Wraps around
                    for pos in range(start, self.genome_length):
                        genome_viz[pos // 2] = 'O'
                    for pos in range(0, end + 1):
                        genome_viz[pos // 2] = 'O'
                
                # Mark 5' and 3' ends based on direction
                if is_clockwise:
                    # CW strand has 5' at start, 3' at end
                    if start < self.genome_length:
                        genome_viz[start // 2] = '5'
                    if end < self.genome_length:
                        genome_viz[end // 2] = '3'
                else:
                    # CCW strand has 3' at start, 5' at end
                    if start < self.genome_length:
                        genome_viz[start // 2] = '3'
                    if end < self.genome_length:
                        genome_viz[end // 2] = '5'
                
                # Mark mutation site if covered and is mutant
                if mutation_status:
                    covers_mutation = (start <= mutation_site <= end) if end >= start else (start <= mutation_site or mutation_site <= end)
                    if covers_mutation:
                        genome_viz[mutation_site // 2] = 'M'
            else:
                # Standard visualization (1 character per nucleotide)
                if end >= start:
                    for pos in range(start, end + 1):
                        genome_viz[pos] = 'O'
                else:  # Wraps around
                    for pos in range(start, self.genome_length):
                        genome_viz[pos] = 'O'
                    for pos in range(0, end + 1):
                        genome_viz[pos] = 'O'
                
                # Mark 5' and 3' ends based on direction
                if is_clockwise:
                    # CW strand has 5' at start, 3' at end
                    if start < self.genome_length:
                        genome_viz[start] = '5'
                    if end < self.genome_length:
                        genome_viz[end] = '3'
                else:
                    # CCW strand has 3' at start, 5' at end
                    if start < self.genome_length:
                        genome_viz[start] = '3'
                    if end < self.genome_length:
                        genome_viz[end] = '5'
                
                # Mark mutation site if covered and is mutant
                if mutation_status:
                    covers_mutation = (start <= mutation_site <= end) if end >= start else (start <= mutation_site or mutation_site <= end)
                    if covers_mutation:
                        genome_viz[mutation_site] = 'M'
            
            # Format compact prefix
            oligo_prefix = "D" if is_duplex else "S"
            direction_symbol = ">" if is_clockwise else "<"
            mut_symbol = "M" if mutation_status else ""
            
            line_prefix = f"{idx} {oligo_prefix}{direction_symbol}{mut_symbol}"
            
            # Calculate percentage for this specific oligo
            oligo_percent = (conc / total_conc * 100) if total_conc > 0 else 0
            
            # Calculate length accounting for circularity
            length = end - start + 1 if end >= start else (self.genome_length - start) + (end + 1)
            
            # Calculate annealing length for duplexes
            ann_len = get_oligo_annealing_length(oligo_type)
            
            # Build detailed information string with fixed-width fields
            info_parts = []
            
            # Add concentration with 2 digits before decimal and 5 after
            # Format to ensure 2 digits before decimal point (e.g., 05.00001 instead of 5.00001)
            info_parts.append(f"{conc:08.5f} uM")
            
            # Add percentage
            info_parts.append(f"{oligo_percent:5.1f}%")
            
            # Add length information - for single stranded, add whitespace for alignment
            if is_duplex:
                if ann_len != "-" and ann_len != "?":
                    info_parts.append(f"    A{ann_len:2d} nt")  # Added padding to match SS length format
                else:
                    info_parts.append(f"    A{ann_len}   nt")  # Added padding to match SS length format
            else:
                info_parts.append(f"     {length:2d} nt")
            
            # Add mutation status information with fixed width
            if is_duplex:
                # Get reactant indices and determine which is CW/CCW
                if len(oligo_type) >= 7:
                    r1_idx, r2_idx = oligo_type[5], oligo_type[6]
                    if (r1_idx is not None and r1_idx < len(self.current_oligomers) and
                        r2_idx is not None and r2_idx < len(self.current_oligomers)):
                        r1 = self.current_oligomers[r1_idx]
                        r2 = self.current_oligomers[r2_idx]
                        r1_is_cw = r1[2]  # Index 2 is is_clockwise
                        r2_is_cw = r2[2]
                        
                        # Determine mutation status with CW/CCW labels
                        if mutation_status == 0:
                            mut_info = "NonMut "
                        elif mutation_status == 3:
                            mut_info = "BothMut"
                        elif mutation_status == 1:  # R1 is mutant
                            mut_info = "CWMut  " if r1_is_cw else "CCWMut "
                        elif mutation_status == 2:  # R2 is mutant
                            mut_info = "CWMut  " if r2_is_cw else "CCWMut "
                        else:
                            mut_info = "UnkMut "
                    else:
                        mut_info = {0: "NonMut ", 1: "R1Mut  ", 2: "R2Mut  ", 3: "BothMut"}.get(mutation_status, "UnkMut ")
                else:
                    mut_info = {0: "NonMut ", 1: "R1Mut  ", 2: "R2Mut  ", 3: "BothMut"}.get(mutation_status, "UnkMut ")
                
                info_parts.append(mut_info)
            else:
                mut_info = "Mutant " if mutation_status else "NonMut "
                info_parts.append(mut_info)
            
            # Add reactant indices for duplexes
            if is_duplex and len(oligo_type) > 6:
                r1_idx, r2_idx = oligo_type[5], oligo_type[6]
                if r1_idx is not None and r2_idx is not None and r1_idx < len(self.current_oligomers) and r2_idx < len(self.current_oligomers):
                    r1 = self.current_oligomers[r1_idx]
                    r2 = self.current_oligomers[r2_idx]
                    r1_start, r1_end = r1[0], r1[1]
                    r2_start, r2_end = r2[0], r2[1]
                    r1_is_cw = r1[2]
                    
                    # Format as "CW:start-end,CCW:start-end"
                    if r1_is_cw:
                        info_parts.append(f"CW:{r1_start}-{r1_end},CCW:{r2_start}-{r2_end}")
                    else:
                        info_parts.append(f"CW:{r2_start}-{r2_end},CCW:{r1_start}-{r1_end}")
                else:
                    info_parts.append(f"({r1_idx},{r2_idx})")
            else:
                info_parts.append(f"{start:2d}-{end:2d}")
            
            # Join all information parts
            info_string = " | ".join(info_parts)
            
            print(f"{line_prefix:>{PREFIX_WIDTH}} " + "".join(genome_viz) + f"  [{info_string}]")
        
        # Show if there are more oligomers not displayed (only count those with meaningful concentrations)
        total_visible_oligos = sum(1 for i, oligo in enumerate(self.current_oligomers)
                                if i < len(self.concentrations) and 
                                self.concentrations[i] > 0 and 
                                self.concentrations[i] != self.DUPLEX_COMPONENT_MARKER)
        

        if total_visible_oligos > len(display_oligos):
            remaining = total_visible_oligos - len(display_oligos)
            print(f"... and {remaining} more oligomers not shown")

