import numpy as np
from vcg_extended_fns import ExtendedOligomerSystem
import sys
from datetime import datetime  # Import datetime module
import vcg_plot_saved_state
import os  # Import os module for directory operations
import analyze_mutant_oligos  # Import the analysis module
import time


class DualOutput:
    """Class to write output to both a file and the console."""
    def __init__(self, file):
        self.file = file

    def write(self, message):
        self.file.write(message)  # Write to the file
        sys.__stdout__.write(message)  # Write to the console

    def flush(self):
        self.file.flush()  # Flush the file buffer


def run_VCG_script(non_stall_frac, plot_dir = 'plots'):
    """
    Test the PCR protocol using a small set of overlapping oligos.
    """
    # Create plots directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)

    #
    # Replace data below to define VCG
    #
            
    genome_length = 60
    min_stable_length = 55
    system = ExtendedOligomerSystem(genome_length)

    oligo_length=25; shift_oligo=5; misaligned_CW_CCW=0; concentration=10.00; global_offset=6;

    oligomers = system.generate_tiling_oligomers(
        oligo_length=oligo_length,
        shift_oligo=shift_oligo,
        misaligned_CW_CCW=misaligned_CW_CCW,
        concentration=concentration,
        global_offset=global_offset
    ) + [(global_offset, global_offset+oligo_length-1, True, 1.0, False, 1, None, None)]

    # Note that even if the last oligo (mutant) in the list does not contain the mutant allele location,
    # subsequent extensions will still likely add the mutant, this is a bug.
    #

    #   No need to change code below this

    #
    #
    #

    # Add all oligomers to the system
    for start, end, is_clockwise, concentration, is_duplex, is_mutant, reactant1_idx, reactant2_idx in oligomers:
        system.add_oligomer_type(start, end, is_clockwise, concentration, is_duplex, is_mutant, reactant1_idx, reactant2_idx)
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("_%H%M_%m%d")  # Format: _HHMM_MMDD
    output_filename = os.path.join(plot_dir, f'vcg_test_script_output{timestamp}.txt')  # Create the output filename
    output_pickle_filename = os.path.join(plot_dir, f'vcg_data{timestamp}.pkl')

    # Redirect output to a text file and console
    with open(output_filename, 'w') as f:


        initial_allele_data = system.compute_allele_concentrations(print_results=False)
        initial_mutant_conc = initial_allele_data['mutant_concentration']
        print(f"Initial mutant concentration: {initial_mutant_conc:.6f}")

        dual_output = DualOutput(f)  # Create an instance of DualOutput
        sys.stdout = dual_output  # Redirect standard output to the dual output

        # Visualize the initial state
        print("\n\n===== INITIAL STATE =====")
        system.print_system_state(visualization_threshold=1e-4, max_oligos=25, multi_track=True, sort_by_concentration=True)  # Customized
        
        # Run PCR protocol WITHOUT de novo nucleation
        print("\n===== RUNNING PCR PROTOCOL =====")
        stats = system.run_pcr_protocol(
            cycles=10,                   # Run 15 PCR cycles
            min_stable_length=min_stable_length,        # Smaller stable length for our small genome
            annealing_time=50.0,         # Shorter annealing time
            cleanup_threshold=1e-5,      # Standard cleanup threshold
            verbose=False,                # Print detailed output
            annealing_verbose=True,    # annealing trajectories plots
            save_plots=False,             # Save visualization plots
            enable_nucleation=False,      # Enable de novo nucleation
            sort_by_concentration=True,    # Visualize oligos sorted by concentration
            min_overlap=2,                  # Minimum overlap for annealing        
            genome_viz_verbose='every_cycle',
            state_save_path = output_pickle_filename,
            stability_threshold = 0.0001, # early termination condition for BDF; % change in concentration
            use_stalling=True,           # Use stalling extension function
            nonstalling_fraction=non_stall_frac,         # Fraction of unstalled strands
            output_dir=plot_dir
        )

        # Print final state
        print("\n===== FINAL STATE =====")
        system.print_system_state(visualization_threshold=1e-4, max_oligos=30, multi_track=True, sort_by_concentration=True)  # Customized
        
        # save the time course
        system.save_system_state(output_pickle_filename)


        final_allele_data = system.compute_allele_concentrations(print_results=False)
        final_mutant_conc = final_allele_data['mutant_concentration']
        print(f"Final mutant concentration: {final_mutant_conc:.6f}")

    sys.stdout = sys.__stdout__  # Reset standard output to console

    output_dir = plot_dir

    # make plots
    vcg_plot_saved_state.plot_from_file(output_pickle_filename, output_dir=output_dir)
    
    # Run mutant oligo analysis
    print("\n===== RUNNING MUTANT OLIGO ANALYSIS =====")
    state_data = analyze_mutant_oligos.load_system_state(output_pickle_filename)
    melted_data = analyze_mutant_oligos.melt_all_duplexes(state_data)

    mutant_analysis = analyze_mutant_oligos.analyze_pcr_cycles(state_data)
    melted_mutant_analysis = analyze_mutant_oligos.analyze_pcr_cycles(melted_data)

    # Extract timestamp from the pickle filename
    timestamp = timestamp  # Use the timestamp already available
    
    analyze_mutant_oligos.plot_results(mutant_analysis, melted_mutant_analysis, output_dir=output_dir, timestamp=timestamp)
    print(f"Mutant oligo analysis complete. Plots saved to {output_dir} directory.")

    return system, stats

for non_stall_frac in [1.0, np.exp(-1.), np.exp(-10.)]:

    print(f"Running simulation with non-stalling fraction = {non_stall_frac}")
    system, stats = run_VCG_script(non_stall_frac, plot_dir='Fig6B_plots')
    time.sleep(60)  # Wait for 1 minute
