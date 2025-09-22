# Code generated with assistance from Cursor (Claude Sonnet 4)

# VCG (Virtual Cell Growth) Simulation Framework

A Python framework for simulating PCR-like protocols with oligomer annealing, extension, and melting dynamics on distributed circular genomes. The system models concentration-based kinetics using ODE integration to study mutant allele amplification and stalling effects.

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
- **`stalling_parameter_sweep.py`** - Explores effects of stalling fraction on mutant amplification
- **`virtualness_parameter_sweep.py`** - Studies shift oligo parameter effects (Fig. 4 A,B,C)
- **`position_parameter_sweep.py`** - Analyzes global offset parameter effects (Fig. 4 D)
- **`stalling_parameter_examples.py`** - Example stalling simulations

### Analysis Tools
- **`analyze_mutant_oligos.py`** - Detailed analysis of mutant oligo binding partners and dynamics
- **`compare_simulations.py`** - Compare final mutant/WT ratios across multiple simulations
- **`vcg_plot_saved_state.py`** - Generate advanced plots from saved simulation states

### Configuration
- **`VCG_library.txt`** - Library of VCG configurations for validation studies


## Dependencies

- **numpy** - Numerical computations (2.1.3)
- **scipy** - ODE integration (`solve_ivp`) (1.15.2)
- **matplotlib** - Plotting and visualization (3.10.1)
- **numba** - JIT compilation for performance (optional, 0.16.0)

## Output Directories

All functions accept customizable output directories with sensible defaults:

- `Fig4ABC_plots/` - Virtualness parameter sweep results  
- `Fig4D_plots/` - Position parameter sweep results
- `Fig6B_plots/` - Stalling example results
- `stall_sweep_plots/` - Stalling parameter sweep results