# VCG (Virtual Circular Genome) Simulation Framework

A Python framework for simulating PCR-like protocols with oligomer annealing, extension, and melting dynamics on distributed circular genomes. The system models concentration-based kinetics using ODE integration to study mutant allele amplification and stalling effects. Code generated with assistance from Cursor (Claude Sonnet 4).

## Core Components

### Main Classes
- **`CoreOligomerSystem`** (`vcg_core_fns.py`) - Base oligomer annealing system with ODE integration
- **`ExtendedOligomerSystem`** (`vcg_extended_fns.py`) - Extended functionality including visualization and state management

### Key Features
- **VCG Cycle Simulation**: Complete cycles with annealing, extension, and melting phases
- **Mutation Analysis**: Tracks mutant vs wildtype allele concentrations
- **Visualization**: Comprehensive plotting of length distributions, mutation distances, and cycle dynamics
- **Parameter Sweeps**: Systematic exploration of experimental conditions

## Scripts

### Parameter Sweep Studies
- **`stalling_parameter_sweep.py`** - Effect of stalling fraction on mutant amplification. python stalling_parameter_sweep.py to run.
- **`virtualness_parameter_sweep.py`** - Effect of genome virtualness on mutant amplification. python virtualness_parameter_sweep.py to run.
- **`position_parameter_sweep.py`** - Effect of initial mutant position on mutant amplification. python position_parameter_sweep.py to run.
- **`stalling_parameter_examples.py`** - Example stalling simulations. python stalling_parameter_examples.py to run.

### Analysis Tools
- **`analyze_mutant_oligos.py`** - Detailed analysis of mutant oligo binding partners and dynamics
- **`compare_simulations.py`** - Compare final mutant/WT ratios across multiple simulations. python compare_simulations.py to run after modifying simulations of interest.
- **`vcg_plot_saved_state.py`** - Generate advanced plots from saved simulation states

### Configuration
- **`VCG_library.txt`** - Library of VCG configurations for validation studies


## Dependencies

- **numpy** - Numerical computations (2.1.3)
- **scipy** - ODE integration (`solve_ivp`) (1.15.2)
- **matplotlib** - Plotting and visualization (3.10.1)
- **numba** - JIT compilation for performance (optional, 0.16.0)

## Output Directories

All functions accept customizable output directories. Defaults as follows:

- `Fig4ABC_plots/` - Virtualness parameter sweep results  
- `Fig4D_plots/` - Position parameter sweep results
- `Fig6B_plots/` - Stalling example results
- `stall_sweep_plots/` - Stalling parameter sweep results
