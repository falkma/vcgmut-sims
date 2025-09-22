import numpy as np
import matplotlib.pyplot as plt
from vcg_extended_fns import ExtendedOligomerSystem
import os
import sys
from datetime import datetime
import pickle
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

def melt_all_duplexes_in_system(system):
    """
    Melt all duplexes in the current system into single strands.
    This replicates the behavior of melt_all_duplexes from vcg_plot_saved_state.py
    but works directly on a live system instead of saved state data.
    
    Args:
        system: ExtendedOligomerSystem instance
        
    Returns:
        None (modifies system in place)
    """
    # Melt all duplexes by setting min_stable_length above genome_length
    # This forces all duplexes to be melted into their constituent single strands
    system.simulate_melt_cycle(min_stable_length=system.genome_length + 1, verbose=False)

def compute_allele_concentrations_like_plot_script(system):
    """
    Compute allele concentrations using the same method as extract_cycle_data_ss 
    from vcg_plot_saved_state.py. This first melts all duplexes, then only 
    considers single-stranded oligos that cover the mutation site.
    
    Args:
        system: ExtendedOligomerSystem instance
        
    Returns:
        Dictionary with mutant_concentration, wildtype_concentration, and ratio
    """
    # First, melt all duplexes just like in vcg_plot_saved_state.py
    melt_all_duplexes_in_system(system)
    
    genome_length = system.genome_length
    mutation_site = genome_length // 2
    
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
    
    # Initialize concentration counters
    mutant_conc = 0.0
    wildtype_conc = 0.0
    
    # Process all oligomers (similar to extract_cycle_data_ss)
    for idx, oligo in enumerate(system.current_oligomers):
        if idx >= len(system.concentrations):
            continue
            
        conc = system.concentrations[idx]
        if conc <= 0:
            continue
                        
        start, end, is_clockwise, is_duplex, is_mutant = oligo[:5]
        
        # Skip duplexes (same as extract_cycle_data_ss)
        # Note: After melting, there should be no duplexes, but keep this check for safety
        if is_duplex:
            continue
        
        # Calculate distance from 3' end to mutation site
        dist = distance_from_3prime(start, end, is_clockwise, mutation_site)
        
        # If oligo covers mutation site, add to appropriate counter
        if dist >= 0:
            if is_mutant:
                mutant_conc += conc
            else:
                wildtype_conc += conc
    
    # Calculate ratio (avoid division by zero)
    ratio = mutant_conc / wildtype_conc if wildtype_conc > 0 else float('inf')
    
    return {
        'mutant_concentration': mutant_conc,
        'wildtype_concentration': wildtype_conc,
        'mutant_to_wildtype_ratio': ratio
    }

def run_simulation_with_nonstalling_fraction(nonstalling_fraction, verbose=False, output_dir='stall_sweep_plots'):
    """
    Run a single PCR simulation with the given nonstalling fraction.
    
    Args:
        nonstalling_fraction: Float from 0 to 1 indicating the fraction of strands that extend under stalling conditions
        verbose: Whether to print detailed output
        output_dir: Directory to save output files (default: 'stall_sweep_plots')
    
    Returns:
        Dictionary containing initial and final mutant concentrations and other metrics
    """
    print(f"Running simulation with nonstalling fraction = {nonstalling_fraction:.1f}")
    
    # Set up the system (copy parameters from run_VCG_script.py)
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
    
    # Add all oligomers to the system
    for start, end, is_clockwise, concentration, is_duplex, is_mutant, reactant1_idx, reactant2_idx in oligomers:
        system.add_oligomer_type(start, end, is_clockwise, concentration, is_duplex, is_mutant, reactant1_idx, reactant2_idx)
    
    # Get initial mutant allele concentration using the plot script method
    initial_allele_data = compute_allele_concentrations_like_plot_script(system)
    initial_mutant_conc = initial_allele_data['mutant_concentration']
    initial_wildtype_conc = initial_allele_data['wildtype_concentration']
    initial_ratio = initial_allele_data['mutant_to_wildtype_ratio']
    
    if verbose:
        print(f"Initial mutant concentration: {initial_mutant_conc:.6f}")
        print(f"Initial wildtype concentration: {initial_wildtype_conc:.6f}")
        print(f"Initial M/W ratio: {initial_ratio:.6f}")
    
    # Create a unique output filename for this nonstalling fraction
    timestamp = datetime.now().strftime("_%H%M_%m%d")
    output_pickle_filename = os.path.join(output_dir, f'vcg_data_stalling_{nonstalling_fraction:.1f}{timestamp}.pkl')
    
    # Redirect output to suppress verbose simulation output
    if not verbose:
        # Create a simple output suppressor
        class DummyOutput:
            def write(self, text):
                pass
            def flush(self):
                pass
        
        original_stdout = sys.stdout
        sys.stdout = DummyOutput()
    
    try:
        # Run PCR protocol with stalling
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
            nonstalling_fraction=nonstalling_fraction,  # Set the nonstalling fraction
            output_dir=output_dir
        )
    finally:
        # Restore stdout
        if not verbose:
            sys.stdout = original_stdout
    
    # Get final mutant allele concentration using the plot script method
    final_allele_data = compute_allele_concentrations_like_plot_script(system)
    final_mutant_conc = final_allele_data['mutant_concentration']
    final_wildtype_conc = final_allele_data['wildtype_concentration']
    final_ratio = final_allele_data['mutant_to_wildtype_ratio']
    
    if verbose:
        print(f"Final mutant concentration: {final_mutant_conc:.6f}")
        print(f"Final wildtype concentration: {final_wildtype_conc:.6f}")
        print(f"Final M/W ratio: {final_ratio:.6f}")
    
    # Calculate growth metrics
    mutant_growth = ((final_mutant_conc)/initial_mutant_conc) #/ ((final_wildtype_conc - initial_wildtype_conc)/initial_wildtype_conc)
    mutant_growth_ratio = final_mutant_conc / initial_mutant_conc if initial_mutant_conc > 0 else float('inf')
    ratio_growth = final_ratio / initial_ratio if initial_ratio > 0 else float('inf')
    
    # Clean up the pickle file to save space
    if os.path.exists(output_pickle_filename):
        os.remove(output_pickle_filename)
    
    print(f"nonstalling fraction {nonstalling_fraction:.1f}: Mutant growth = {mutant_growth:.6f} (ratio = {mutant_growth_ratio:.2f}x)")
    
    return {
        'nonstalling_fraction': nonstalling_fraction,
        'initial_mutant_conc': initial_mutant_conc,
        'final_mutant_conc': final_mutant_conc,
        'initial_wildtype_conc': initial_wildtype_conc,
        'final_wildtype_conc': final_wildtype_conc,
        'initial_ratio': initial_ratio,
        'final_ratio': final_ratio,
        'mutant_growth': mutant_growth,
        'mutant_growth_ratio': mutant_growth_ratio,
        'ratio_growth': ratio_growth,
        'final_cycle_stats': stats[-1] if stats else None
    }

def run_parameter_sweep(verbose_individual=False, verbose_summary=True, output_dir='stall_sweep_plots'):
    """
    Run parameter sweep across nonstalling fractions from 0.0 to 1.0 in steps of 0.1.
    
    Args:
        verbose_individual: Whether to print verbose output for individual simulations
        verbose_summary: Whether to print summary results
        output_dir: Directory to save output files (default: 'stall_sweep_plots')
    
    Returns:
        List of dictionaries containing results for each nonstalling fraction
    """
    # Create plots directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define nonstalling fractions to test
    #nonstalling_fractions = np.exp(-np.linspace(0, 10, 21)) 
    #nonstalling_fractions = 1 - np.exp(-np.linspace(0, 10, 21))
    nonstalling_fractions = 1 - np.exp(-1/np.linspace(0, 30, 31))
    #nonstalling_fractions = np.linspace(0, 1, 21)

    print(f"Starting parameter sweep with {len(nonstalling_fractions)} nonstalling fraction values")
    print(f"nonstalling fractions: {[f'{sf:.1f}' for sf in nonstalling_fractions]}")
    
    results = []
    
    # Run simulations for each nonstalling fraction
    for i, nonstalling_fraction in enumerate(nonstalling_fractions):
        print(f"\n=== SIMULATION {i+1}/{len(nonstalling_fractions)} ===")
        
        try:
            result = run_simulation_with_nonstalling_fraction(
                nonstalling_fraction, 
                verbose=verbose_individual,
                output_dir=output_dir
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR: Simulation failed for nonstalling fraction {nonstalling_fraction:.1f}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if verbose_summary:
        print(f"\n=== PARAMETER SWEEP COMPLETE ===")
        print(f"Successfully completed {len(results)}/{len(nonstalling_fractions)} simulations")
        
        # Print summary table
        print(f"\n{'Nonstalling':<10} {'Init Mut':<10} {'Final Mut':<11} {'Growth':<10} {'Growth Ratio':<12} {'M/W Ratio Growth':<15}")
        print(f"{'Fraction':<10} {'Conc':<10} {'Conc':<11} {'(Abs)':<10} {'(Fold)':<12} {'(Fold)':<15}")
        print("-" * 80)
        
        for result in results:
            sf = result['nonstalling_fraction']
            init_mut = result['initial_mutant_conc']
            final_mut = result['final_mutant_conc']
            growth = result['mutant_growth']
            growth_ratio = result['mutant_growth_ratio']
            ratio_growth = result['ratio_growth']
            
            print(f"{sf:<10.1f} {init_mut:<10.6f} {final_mut:<11.6f} {growth:<10.6f} {growth_ratio:<12.2f} {ratio_growth:<15.2f}")
    
    return results

def plot_parameter_sweep_results(results, output_dir='stall_sweep_plots'):
    """
    Create plots showing mutant allele concentration growth as a function of nonstalling fraction.
    
    Args:
        results: List of result dictionaries from run_parameter_sweep
        output_dir: Directory to save plots
    """
    if len(results) == 0:
        print("No results to plot!")
        return
    
    # Extract data for plotting
    nonstalling_fractions = np.array([r['nonstalling_fraction'] for r in results])
    mutant_growths = np.array([r['mutant_growth'] for r in results])
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("_%H%M_%m%d")
    
    # Create a single stretched out plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle('Mutant Allele Growth vs stalling factor', fontsize=16, fontweight='bold')
    
    # Plot: Absolute Growth in Mutant Concentration
    ax.plot(-1/np.log(1-nonstalling_fractions), mutant_growths, 'o-', color='#fbb040', linewidth=2)
    #ax.plot(nonstalling_fractions, mutant_growths, 'o-', color='#fbb040', linewidth=2)
    #ax.plot(-np.log(1-nonstalling_fractions), mutant_growths, 'o-', color='#fbb040', linewidth=2)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('stalling factor')
    ax.set_ylabel('Mutant Concentration Growth (Absolute)')
    ax.set_title('Absolute Growth in Mutant Concentration')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.98, 1.62)
    ax.set_xlim(-1, 31)
    
    plt.tight_layout()
    
    # Save the plot as both PNG and PDF
    png_filename = os.path.join(output_dir, f'stalling_parameter_sweep{timestamp}.png')
    pdf_filename = os.path.join(output_dir, f'stalling_parameter_sweep{timestamp}.pdf')
    
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_filename, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved as PNG: {png_filename}")
    print(f"Plot saved as PDF: {pdf_filename}")
    
    plt.show()
    
    return png_filename, pdf_filename

def main(output_dir='stall_sweep_plots'):
    """
    Main function to run the parameter sweep and create plots.
    
    Args:
        output_dir: Directory to save output files (default: 'stall_sweep_plots')
    """
    print("=== nonstalling fraction PARAMETER SWEEP ===")
    print("This script will test nonstalling fractions from 0.0 to 1.0 in steps of 0.1")
    print("and measure the growth in mutant allele concentration.\n")
    
    # Run the parameter sweep
    results = run_parameter_sweep(verbose_individual=False, verbose_summary=True, output_dir=output_dir)
    
    if len(results) == 0:
        print("ERROR: No successful simulations completed!")
        return
    
    # Create plots
    print(f"\n=== CREATING PLOTS ===")
    png_filename, pdf_filename = plot_parameter_sweep_results(results, output_dir=output_dir)
    
    # Save results to a file
    timestamp = datetime.now().strftime("_%H%M_%m%d")
    results_filename = os.path.join(output_dir, f'stalling_sweep_results{timestamp}.pkl')
    
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results data saved to: {results_filename}")
    
    # Final summary
    print(f"\n=== FINAL SUMMARY ===")
    if len(results) > 0:
        # Find the optimal nonstalling fraction
        mutant_growths = [r['mutant_growth'] for r in results]
        max_growth_idx = np.argmax(mutant_growths)
        optimal_result = results[max_growth_idx]
        
        print(f"Optimal nonstalling fraction: {optimal_result['nonstalling_fraction']:.1f}")
        print(f"Maximum mutant growth: {optimal_result['mutant_growth']:.6f}")
        print(f"Growth ratio at optimum: {optimal_result['mutant_growth_ratio']:.2f}x")
        print(f"Final M/W ratio at optimum: {optimal_result['final_ratio']:.4f}")
        
        # Print range of effects
        min_growth = min(mutant_growths)
        max_growth = max(mutant_growths)
        print(f"\nRange of mutant growth: {min_growth:.6f} to {max_growth:.6f}")
        print(f"Effect of nonstalling fraction: {(max_growth - min_growth):.6f} absolute difference")
    
    print(f"\nParameter sweep complete! Check the {output_dir} directory for results.")

if __name__ == "__main__":
    main() 