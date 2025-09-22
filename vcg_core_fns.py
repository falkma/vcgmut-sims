import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from datetime import datetime
import sys
import os

# Add Numba import
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    print("Numba is available and will be used for acceleration")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba is not available - falling back to standard Python")
 
class CoreOligomerSystem:
    """A concentration-based oligomer annealing system using ODE integration"""
    def __init__(self, genome_length):
        """
        Initialize the system with a circular genome.
        
        Args:
            genome_length: Length of the underlying circular genome
        """
        self.genome_length = genome_length
        self.current_oligomers = []  # List of (start, end, is_clockwise, is_duplex, is_mutant, reactant1_idx, reactant2_idx) tuples
        self.oligomer_indices = {}  # Maps oligomer type to its index in the concentration vector
        self.concentrations = None  # Will be a numpy array of concentrations
        self.reaction_matrix = None  # Stoichiometric matrix for reactions
        self.reaction_pairs = []  # List of (i, j, k) indices for reactants and product
        
        # Track which oligomers have been involved in reaction network building
        self.processed_oligomers = set()
        
        # Track cycle history
        self.cycle_history = {
            'total_oligos': [],
            'single_stranded': [],
            'duplexes': [],
            'unique_products': set(),
            'mutant_oligos': [],
            'duplexes_by_mutation': []
        }

        # Special concentration value for oligomers that are part of duplexes but not present as single strands
        self.DUPLEX_COMPONENT_MARKER = -1.0

    def add_oligomer_type(self, start, end, is_clockwise, concentration=0.0, is_duplex=False, is_mutant=0, reactant1_idx=None, reactant2_idx=None):
        """
        Add an oligomer type to the system.
        
        Args:
            start: Start position (0-indexed)
            end: End position (0-indexed)
            is_clockwise: Direction (True for clockwise, False for counterclockwise)
            concentration: Initial concentration (use -1 for components of duplexes with no free form)
            is_duplex: Whether this oligomer is a duplex (product of annealing)
            is_mutant: Whether this oligomer has a mutation at site L/2
            reactant1_idx, reactant2_idx: Indices of reactants that formed this oligomer (for duplexes)
            
        Returns:
            Index of the new oligomer type
        """
        # For duplexes, reactant indices must be provided
        if is_duplex and (reactant1_idx is None or reactant2_idx is None):
            raise ValueError("Reactant indices must be provided for duplexes")
        
        # For non-duplexes, reactant indices should be None
        if not is_duplex:
            reactant1_idx = None
            reactant2_idx = None
        
        # Create the complete oligomer tuple
        oligo_type = (start, end, is_clockwise, is_duplex, is_mutant, reactant1_idx, reactant2_idx)
        
        # Check if this exact oligomer already exists (with all 7 elements)
        if oligo_type in self.oligomer_indices:
            idx = self.oligomer_indices[oligo_type]
            
            # Update concentration if needed
            if self.concentrations is not None:
                # If the oligomer already exists but has the special DUPLEX_COMPONENT_MARKER value (-1)
                # and we're adding a non-negative concentration, replace the marker with the actual concentration
                if self.concentrations[idx] == self.DUPLEX_COMPONENT_MARKER and concentration >= 0.0:
                    self.concentrations[idx] = concentration
                # If it's a regular oligomer, add to existing concentration
                elif self.concentrations[idx] >= 0.0 and concentration >= 0.0:
                    self.concentrations[idx] += concentration
                # If we're adding a DUPLEX_COMPONENT_MARKER but the oligomer already has a positive concentration,
                # keep the positive concentration and ignore the marker
                elif concentration == self.DUPLEX_COMPONENT_MARKER and self.concentrations[idx] >= 0.0:
                    pass  # Keep existing concentration
            
            return idx
        
        # For non-duplexes, we can check for a match on just the first 5 elements
        if not is_duplex:
            # Look for an exact match on the base properties for non-duplexes
            for idx, existing_oligo in enumerate(self.current_oligomers):
                if (not existing_oligo[3] and  # Not a duplex
                    existing_oligo[0] == start and
                    existing_oligo[1] == end and
                    existing_oligo[2] == is_clockwise and
                    existing_oligo[4] == is_mutant):
                    
                    # Update concentration
                    if self.concentrations is not None:
                        if self.concentrations[idx] == self.DUPLEX_COMPONENT_MARKER and concentration >= 0.0:
                            self.concentrations[idx] = concentration
                        elif self.concentrations[idx] >= 0.0 and concentration >= 0.0:
                            self.concentrations[idx] += concentration
                        elif concentration == self.DUPLEX_COMPONENT_MARKER and self.concentrations[idx] >= 0.0:
                            pass
                    
                    return idx
        
        # If we get here, we need to add a new oligomer
        index = len(self.current_oligomers)
        self.current_oligomers.append(oligo_type)
        self.oligomer_indices[oligo_type] = index
        
        # Update concentrations array if it already exists
        if self.concentrations is not None:
            # Append the new concentration to the array
            self.concentrations = np.append(self.concentrations, concentration)
        else:
            # Initialize concentrations array
            self.concentrations = np.zeros(index + 1)
            self.concentrations[index] = concentration
        
        return index

    def build_reaction_network(self, rebuild=False, verbose=False, min_overlap=3):
        """
        Build the reaction network for the current set of oligomers.
        Optimized version for very large systems (10k+ oligomers, 70k+ reactions).
        Uses only analytical Jacobian for efficient ODE integration.
        
        Args:
            rebuild: Whether to rebuild the network from scratch
            verbose: Whether to print detailed output
            conc_threshold: Concentration threshold for including oligomers (deprecated, kept for backwards compatibility)
            min_overlap: Minimum overlap required for annealing
                
        Returns:
            Number of reactions in the network
        """
        import time
        start_time = time.time()
        
        if rebuild:
            # Clear existing reaction network
            self.reaction_matrix = None
            self.reaction_pairs = []
            self.processed_oligomers = set()
            # Clear the Jacobian
            self._jac_sparsity = None
        
        n_oligomers = len(self.current_oligomers)
        if n_oligomers == 0:
            return 0
            
        # Filter out only DUPLEX_COMPONENT_MARKER oligomers, no concentration threshold
        if self.concentrations is not None:
            valid_indices = np.where(self.concentrations != self.DUPLEX_COMPONENT_MARKER)[0]
            valid_indices = valid_indices[valid_indices < n_oligomers]
            if len(valid_indices) == 0:
                max_idx = 0
            else:
                max_idx = np.max(valid_indices)
        else:
            valid_indices = np.arange(n_oligomers)
            max_idx = n_oligomers - 1
        
        # Initialize data structures for storing reactions efficiently
        new_reactions = 0
        
        # Use sets for faster lookup of existing reactions
        existing_reaction_pairs = set()
        for i, j, _ in self.reaction_pairs:
            existing_reaction_pairs.add((i, j))
            existing_reaction_pairs.add((j, i))
        
        if not self.processed_oligomers:
            print("Building entire reaction network...")
        else:
            print("Updating reaction network with new oligomers...")
        
        if verbose:
            print(f"System currently has {len(self.reaction_pairs)} reaction pairs and {len(self.concentrations)} concentrations")
        
        # Determine which oligomers need to be processed - use numpy for efficiency
        unprocessed = np.array([i for i in valid_indices if i not in self.processed_oligomers])
        
        # Add all unprocessed oligomers to processed set (we'll process them now)
        self.processed_oligomers.update(unprocessed)
        
        if verbose:
            print(f"Processing {len(unprocessed)} new oligomers among {n_oligomers} total")
        
        # Create boolean mask for faster lookups of unprocessed oligomers
        # This replaces expensive set membership checks with fast array indexing
        max_oligo_idx = max(max_idx, np.max(unprocessed) if len(unprocessed) > 0 else 0)
        is_unprocessed = np.zeros(max_oligo_idx + 1, dtype=bool)
        is_unprocessed[unprocessed] = True
        
        # Pre-compute oligomer properties for better efficiency - focus only on valid oligomers
        # Use structured NumPy arrays for better memory and cache efficiency
        oligomer_props = []
        
        # Define the dtype for better memory layout and vectorized operations
        props_dtype = np.dtype([
            ('idx', np.int32),
            ('start', np.int32),
            ('end', np.int32),
            ('is_clockwise', np.bool_),
            ('is_duplex', np.bool_),
            ('is_mutant', np.int8)
        ])
        
        for i in valid_indices:
            if i >= n_oligomers:
                continue
                
            oligo = self.current_oligomers[i]
            start, end, is_clockwise, is_duplex, is_mutant = oligo[:5]
            
            # Skip duplexes - they can't anneal
            if is_duplex:
                continue
                
            # Only skip oligomers with DUPLEX_COMPONENT_MARKER, no concentration threshold
            if (self.concentrations is not None and
                (i >= len(self.concentrations) or 
                 self.concentrations[i] == self.DUPLEX_COMPONENT_MARKER)):
                continue
                
            # Add to the list
            oligomer_props.append((i, start, end, is_clockwise, is_duplex, is_mutant))
        
        # Convert to structured array for better performance
        if oligomer_props:
            oligo_array = np.array(oligomer_props, dtype=props_dtype)
            
            # Quick filtering: Group by direction for more efficient pair generation
            clockwise_mask = oligo_array['is_clockwise']
            clockwise_oligos = oligo_array[clockwise_mask]
            counter_oligos = oligo_array[~clockwise_mask]
            
            if verbose:
                print(f"Found {len(clockwise_oligos)} clockwise and {len(counter_oligos)} counterclockwise oligomers")
        else:
            # No valid oligomers
            if verbose:
                print("No valid oligomers found for reaction network")
            return len(self.reaction_pairs)
        
        # For sparse matrix construction, prepare data structures
        rows, cols, data = [], [], []
        new_reaction_pairs = []
        
        # Generate all possible clockwise-counterclockwise pairs efficiently
        # This is the most time-consuming part, so optimize it heavily
        pairs_to_process = []
        
        # Process all pairs between unprocessed and all others
        for cw_oligo in clockwise_oligos:
            i = cw_oligo['idx']
            for ccw_oligo in counter_oligos:
                j = ccw_oligo['idx']
                
                # Skip if already processed
                if (i, j) in existing_reaction_pairs or (j, i) in existing_reaction_pairs:
                    continue
                    
                # At least one must be unprocessed - optimized check using boolean array
                if not (is_unprocessed[i] or is_unprocessed[j]):
                    continue
                
                pairs_to_process.append((cw_oligo, ccw_oligo))
        
        #if verbose:
            #print(f"Checking {len(pairs_to_process)} potential reaction pairs")
        
        # Process all potential reaction pairs
        for cw_oligo, ccw_oligo in pairs_to_process:
            i, start1, end1, _, _, is_mutant1 = cw_oligo
            j, start2, end2, _, _, is_mutant2 = ccw_oligo
            
            # Create simplified tuples for can_anneal
            oligo1 = (start1, end1, True, False, is_mutant1)
            oligo2 = (start2, end2, False, False, is_mutant2)
            
            # Check if they can anneal
            can_anneal_result, _ = self.can_anneal(oligo1, oligo2, min_overlap)
            
            if can_anneal_result:
                # Calculate the product using the anneal_product function
                product = self.anneal_product(oligo1, oligo2, verbose=False)
                
                if product is None:
                    continue
                
                # Verify the product is marked as a duplex
                if not product[3]:
                    if verbose:
                        print(f"WARNING: Product from anneal_product not marked as duplex: {product}")
                    # Force it to be a duplex
                    product = (product[0], product[1], product[2], True, product[4])
                
                # Extend product with reactant indices
                product_with_reactants = product + (i, j)
                
                # Get or create the product's index
                k = self.add_oligomer_type(
                    product_with_reactants[0],  # start
                    product_with_reactants[1],  # end
                    product_with_reactants[2],  # is_clockwise (always True after standardization)
                    0.0,                        # concentration
                    product_with_reactants[3],  # is_duplex (should be True)
                    product_with_reactants[4],  # mutation_code
                    i,                          # reactant1_idx
                    j                           # reactant2_idx
                )
                
                # Store reactants at DUPLEX_COMPONENT_MARKER concentration if needed
                if self.concentrations[i] == 0:
                    self.concentrations[i] = self.DUPLEX_COMPONENT_MARKER
                if self.concentrations[j] == 0:
                    self.concentrations[j] = self.DUPLEX_COMPONENT_MARKER
                
                # Create sparse matrix entries directly (-1 for reactants, +1 for product)
                rows.extend([i, j, k])
                cols.extend([new_reactions, new_reactions, new_reactions])
                data.extend([-1, -1, 1])
                
                # Track the reaction pair
                new_reaction_pairs.append((i, j, k))
                existing_reaction_pairs.add((i, j))
                existing_reaction_pairs.add((j, i))
                
                new_reactions += 1
                
                

        # Build the sparse reaction matrix efficiently
        if new_reactions > 0:
            from scipy import sparse
            
            # Calculate the required dimensions
            #max_idx_for_matrix = max(max_idx, np.max([max(i, j, k) for i, j, k in new_reaction_pairs]) + 1)
            max_idx_for_matrix = len(self.concentrations)
            #print("Here are all the new reaction pairs: ", new_reaction_pairs)

            print("Max index = ", max_idx_for_matrix," but we have concentrations for ", len(self.concentrations))
            # Create new sparse matrix for the added reactions
            new_stoich_matrix = sparse.coo_matrix(
                (data, (rows, cols)), 
                shape=(max_idx_for_matrix, new_reactions)
            ).tocsr()
            
            if self.reaction_matrix is None:
                # First time building the matrix
                self.reaction_matrix = new_stoich_matrix
            else:
                # Convert existing matrix to sparse if it's not already
                if not isinstance(self.reaction_matrix, sparse.spmatrix):
                    self.reaction_matrix = sparse.csr_matrix(self.reaction_matrix)
                
                
                # Append new columns efficiently
                self.reaction_matrix = sparse.hstack([self.reaction_matrix, new_stoich_matrix])
            
        # Store references to reactions for easy access
        self.reaction_pairs.extend(new_reaction_pairs)
        
        # Convert to numpy array for faster lookup during simulation
        self._reaction_pairs_array = np.array(self.reaction_pairs)
        
        # Convert reaction matrix to CSR format for faster operations
        if isinstance(self.reaction_matrix, sparse.spmatrix) and not isinstance(self.reaction_matrix, sparse.csr_matrix):
            self.reaction_matrix = self.reaction_matrix.tocsr()
        
        # Store CSR matrix components for numba-accelerated computations
        if isinstance(self.reaction_matrix, sparse.csr_matrix):
            self._csr_matrix_data = self.reaction_matrix.data
            self._csr_matrix_indices = self.reaction_matrix.indices
            self._csr_matrix_indptr = self.reaction_matrix.indptr
            self._csr_matrix_shape = self.reaction_matrix.shape
        

        if verbose:
            print("Dimensions of reaction matrix =", self.reaction_matrix.shape)
            print(f"Added {new_reactions} reactions by checking {len(pairs_to_process)} potential reaction pairs.")

        if verbose:
            print(f"System now has {len(self.reaction_pairs)} reactions that involve {len(self.current_oligomers)} potential species (ss + duplexes). System has{len(self.concentrations)} concentrations.")
            print(f"Reaction network built in {time.time() - start_time:.2f} seconds")

        
        # Set up the analytical Jacobian after all reaction data is updated
        #if new_reactions > 0:
            #print("\n Setting up analytical Jacobian...")
            #self._setup_analytical_jacobian()
        
        # 
        # 
        # Code to report which oligos are not participating in any reactions
        #
        #

        if verbose:
            # Find single-stranded oligomers that participate in some reaction
            involved_indices = set(i for i, j, _ in self.reaction_pairs).union(j for i, j, _ in self.reaction_pairs)        
            # Get inactive single-stranded oligos using get_single_stranded_oligos
            # This method returns a list of (oligo_type, concentration) tuples
            inactive_oligos = []
            for i, oligo_type in enumerate(self.current_oligomers):
                if (not oligo_type[3] and  # Not a duplex
                    i < len(self.concentrations) and 
                    self.concentrations[i] >= 0 and 
                    i not in involved_indices):
                    inactive_oligos.append((i, oligo_type))
                    
            if inactive_oligos:
                print("\n----- Single-Stranded Oligomers Not Participating in Any Reactions -----")
                print(f"Found {len(inactive_oligos)} ss oligomers with conc >0.1 that don't participate in reactions:")
                for idx, oligo_type in inactive_oligos:
                    print(f"  SS {idx}: {oligo_type[0]}-{oligo_type[1]} ({'CW' if oligo_type[2] else 'CCW'}), "
                        f"Conc: {self.concentrations[idx]:.6f}")
        
        return len(self.reaction_pairs)

    def calculate_overlap(self, oligo1, oligo2):
        """
        Calculate the overlap between two oligomers using efficient BitArrays.
        Returns both the overlap positions (as a bit array) and the overlap size.
        
        Args:
            oligo1: Tuple (start, end, is_clockwise, is_duplex, is_mutant, [reactant1_idx, reactant2_idx]) for first oligomer
            oligo2: Tuple (start, end, is_clockwise, is_duplex, is_mutant, [reactant1_idx, reactant2_idx]) for second oligomer
                
        Returns:
            Tuple (overlap_bit_array, overlap_size) where overlap_bit_array is a boolean NumPy array
        """
        # Extract the relevant parts
        start1, end1 = oligo1[0], oligo1[1]
        start2, end2 = oligo2[0], oligo2[1]
        
        # Initialize bit arrays (optimization: reuse to avoid allocation)
        if not hasattr(self, '_bit_array1'):
            self._bit_array1 = np.zeros(self.genome_length, dtype=bool)
            self._bit_array2 = np.zeros(self.genome_length, dtype=bool)
        else:
            # Reset arrays
            self._bit_array1.fill(False)
            self._bit_array2.fill(False)
        
        # Fill bit arrays
        # Handle circular genome
        if end1 >= start1:
            self._bit_array1[start1:end1+1] = True
        else:  # Wraps around
            self._bit_array1[start1:] = True
            self._bit_array1[:end1+1] = True
            
        if end2 >= start2:
            self._bit_array2[start2:end2+1] = True
        else:  # Wraps around
            self._bit_array2[start2:] = True
            self._bit_array2[:end2+1] = True
        
        # Calculate overlap using fast bitwise AND
        overlap = self._bit_array1 & self._bit_array2
        overlap_size = np.sum(overlap)
        
        return overlap, overlap_size

    def can_anneal(self, oligo1, oligo2, min_overlap=3):
        """
        Check if two oligomer types can anneal, accounting for the circular nature of the genome.
        Optimized with BitArrays for faster overlap calculation.
        
        Args:
            oligo1: Tuple (start, end, is_clockwise, is_duplex, is_mutant, [reactant1_idx, reactant2_idx]) for first oligomer
            oligo2: Tuple (start, end, is_clockwise, is_duplex, is_mutant, [reactant1_idx, reactant2_idx]) for second oligomer
            min_overlap: Minimum required overlap
                
        Returns:
            Boolean indicating whether they can anneal and the overlap size
        """
        # Memoization: Check if we've computed this pair before
        # Create a cache key using only the parts that affect annealing
        cache_key = (
            oligo1[0], oligo1[1], oligo1[2], oligo1[3],
            oligo2[0], oligo2[1], oligo2[2], oligo2[3],
            min_overlap
        )
        
        # Check if we have a memoization cache
        if not hasattr(self, '_anneal_cache'):
            self._anneal_cache = {}
        elif cache_key in self._anneal_cache:
            return self._anneal_cache[cache_key]
        
        # Extract the relevant parts
        start1, end1, is_clockwise1, is_duplex1, _ = oligo1[:5]
        start2, end2, is_clockwise2, is_duplex2, _ = oligo2[:5]
        
        # Early rejection tests (avoid expensive calculations)
        # Duplexes cannot anneal again
        if is_duplex1 or is_duplex2:
            self._anneal_cache[cache_key] = (False, 0)
            return False, 0
        
        # Oligomers must have opposite directions to anneal
        if is_clockwise1 == is_clockwise2:
            self._anneal_cache[cache_key] = (False, 0)
            return False, 0
        
        # Use centralized overlap calculation
        _, overlap_size = self.calculate_overlap(oligo1, oligo2)
        
        # Cache the result
        result = (overlap_size >= min_overlap, overlap_size)
        self._anneal_cache[cache_key] = result
        
        return result

    def anneal_product(self, oligo1, oligo2, verbose=False, min_overlap=3):
        """
        Calculate the product of annealing two oligomer types,
        accounting for the circular nature of the genome.
        Standardizes all duplexes to be clockwise.
        Optimized with BitArrays for faster calculations.
        
        Args:
            oligo1: Tuple (start, end, is_clockwise, is_duplex, is_mutant, [reactant1_idx, reactant2_idx]) for first oligomer
            oligo2: Tuple (start, end, is_clockwise, is_duplex, is_mutant, [reactant1_idx, reactant2_idx]) for second oligomer
            verbose: Whether to print detailed output
            min_overlap: Minimum overlap required for annealing
                
        Returns:
            Tuple representing the product oligomer with encoded mutation status
            (does NOT include reactant indices - those are added later)
        """
        # First check if they can anneal with the specified min_overlap
        can_anneal_result, overlap_size = self.can_anneal(oligo1, oligo2, min_overlap)
        if not can_anneal_result:
            if verbose:
                print(f"Cannot anneal oligomers: insufficient overlap ({overlap_size} < {min_overlap})")
            return None
            
        # Extract only the first 5 elements, ignoring any reactant indices
        start1, end1, is_clockwise1, _, is_mutant1 = oligo1[:5]
        start2, end2, is_clockwise2, _, is_mutant2 = oligo2[:5]
        
        # Use the centralized overlap calculation method
        bit_array1, bit_array2 = self._bit_array1, self._bit_array2  # These are already set by can_anneal through calculate_overlap
        
        # Combine positions with efficient bitwise OR
        all_positions = bit_array1 | bit_array2
        
        # Convert to list of indices efficiently
        pos_list = np.where(all_positions)[0]
        
        # Default start/end for full genome coverage
        if len(pos_list) >= self.genome_length:
            new_start, new_end = 0, self.genome_length - 1
        else:
            # Find largest gap more efficiently using numpy operations
            pos_with_wrap = np.append(pos_list, pos_list[0] + self.genome_length)
            gaps = np.diff(pos_with_wrap)
            
            if len(gaps) > 0:
                largest_gap_idx = np.argmax(gaps) 
                
                if largest_gap_idx < len(pos_list) - 1:
                    # Regular gap (not wrap-around)
                    new_start = pos_list[largest_gap_idx + 1]
                    new_end = pos_list[largest_gap_idx]
                else:
                    # Wrap-around gap
                    new_start = pos_list[0]
                    new_end = pos_list[-1]
            else:
                # Single position
                new_start = new_end = pos_list[0]
        
        # Calculate mutation code
        mutation_code = (1 if is_mutant1 else 0) + (2 if is_mutant2 else 0)
        
        # Standardize direction to always be clockwise
        is_clockwise = True
        
        # Print debug info
        if verbose:
            print(f"Annealing {oligo1} and {oligo2}")
            print(f"Created standardized clockwise duplex: ({new_start}-{new_end})")
        
        # Return product (only 5 elements, without reactant indices)
        # Reactant indices will be added when this is used in build_reaction_network
        return (new_start, new_end, is_clockwise, True, mutation_code)

    def extension_oligos(self, time=100.0, verbose=False, min_overlap=3):
        """
        Simulate extension of oligomers during PCR.
        
        Args:
            time: Time for extension
            verbose: Whether to print detailed output
            min_overlap: Minimum overlap required for annealing
                
        Returns:
            None
        """
        # No need to rebuild reaction network here - it's already built during annealing phase
        # and not used by the extension logic. This avoids redundant Jacobian setup.
        
        # Get all duplexes regardless of concentration
        duplexes = self.get_duplexes(0)
        if not duplexes:
            print("No duplexes found for extension")
            return {}
        
        print(f"Simulating extension for {len(duplexes)} duplex types...")
        
        # Mutation site (midpoint of genome)
        mutation_site = self.genome_length // 2
        
        # Create a dictionary for extended duplexes (to return)
        extended_oligos = {}
        
        # Initialize bit arrays for position tracking (reuse instead of recreating)
        if not hasattr(self, '_extension_bit_array1'):
            self._extension_bit_array1 = np.zeros(self.genome_length, dtype=bool)
            self._extension_bit_array2 = np.zeros(self.genome_length, dtype=bool)
        
        # Helper function to calculate length considering circular genome
        def calc_length(start, end):
            if end >= start:
                return end - start + 1
            else:
                return (self.genome_length - start) + end + 1
        
        # Optimized position checking using bit arrays
        def fill_positions(bit_array, start, end):
            bit_array.fill(False)  # Reset array
            if end >= start:
                bit_array[start:end+1] = True
            else:  # Wraps around
                bit_array[start:] = True
                bit_array[:end+1] = True
            return bit_array
        
        # Process each duplex
        for i, (duplex_idx, duplex_tuple, conc) in enumerate(duplexes):
            # Unpack components
            dup_start, dup_end, dup_direction, _, dup_mutation = duplex_tuple[:5]
            
            # Get reactant indices directly from the duplex tuple
            if len(duplex_tuple) >= 7:
                reactant1_idx, reactant2_idx = duplex_tuple[5], duplex_tuple[6]
            else:
                # This shouldn't happen with the updated get_duplexes, but as a fallback:
                if verbose:
                    print(f"  WARNING: Duplex missing reactant indices - this shouldn't happen")
                continue
            
            # Check if these indices are valid
            if (reactant1_idx is None or reactant1_idx >= len(self.current_oligomers) or
                reactant2_idx is None or reactant2_idx >= len(self.current_oligomers)):
                if verbose:
                    print(f"  WARNING: Invalid reactant indices {reactant1_idx}, {reactant2_idx}")
                continue
                
            # Get the reactant tuples
            reactant1 = self.current_oligomers[reactant1_idx]
            reactant2 = self.current_oligomers[reactant2_idx]
            
            r1_start, r1_end, r1_direction, _, r1_mutation = reactant1[:5]
            r2_start, r2_end, r2_direction, _, r2_mutation = reactant2[:5]
            
            # Verify the duplex is clockwise (standard orientation)
            if not dup_direction:
                if verbose:
                    print(f"Warning: Found counterclockwise duplex. All duplexes should be standardized to clockwise direction.")
                # Continue processing anyway for robustness
            
            if verbose:
                print(f"\nExtension for duplex {i+1}: {duplex_tuple}, conc: {conc:.2f}")
            
            # Fill bit arrays with positions (reuse arrays instead of creating sets)
            positions1_array = fill_positions(self._extension_bit_array1, r1_start, r1_end)
            positions2_array = fill_positions(self._extension_bit_array2, r2_start, r2_end)
            
            # Track if any extensions occurred and what the new boundaries are
            extensions_made = False
            new_r1_start, new_r1_end = r1_start, r1_end
            new_r2_start, new_r2_end = r2_start, r2_end
            
            # Check if the oligos cover the mutation site before extension
            r1_covers_mutation = positions1_array[mutation_site]
            r2_covers_mutation = positions2_array[mutation_site]
            
            # Check and extend the 3' end of the first oligo
            r1_extended = False
            if r1_direction:  # Clockwise -> 3' end is at r1_end
                r1_3prime_pos = r1_end
                # Check if there's sufficient overlap (min_overlap) at the 3' end
                # Just need to check if the furthest position is in the partner oligo
                min_overlap_pos = (r1_3prime_pos - (min_overlap - 1)) % self.genome_length
                r1_3prime_overlap_sufficient = positions2_array[min_overlap_pos]
                
                # Combine condition with sufficient overlap check
                if positions2_array[r1_3prime_pos] and r1_3prime_overlap_sufficient and not r2_direction and r1_3prime_pos != r2_end:
                    if verbose:
                        print(f"  Extending r1 (CW) 3' end at {r1_3prime_pos} to r2's 5' end at {r2_end}")
                    
                    # 1. Simply extend to partner's 5' end (r2_end)
                    candidate_r1_end = r2_end
                    
                    # 2. Check if the length increases after extension
                    original_length = calc_length(r1_start, r1_end)
                    new_length = calc_length(r1_start, candidate_r1_end)
                    
                    if new_length > original_length:
                        # Length increased, proceed with extension
                        new_r1_end = candidate_r1_end
                    else:
                        # 3. Length decreased or remained same, set 3' end to one less than the strand's own 5' end
                        new_r1_end = (r1_start - 1) % self.genome_length
                    
                    # Only set flags if an actual extension occurred
                    if new_r1_end != r1_end:
                        extensions_made = True
                        r1_extended = True
            else:  # Counterclockwise -> 3' end is at r1_start
                r1_3prime_pos = r1_start
                # Check if there's sufficient overlap (min_overlap) at the 3' end
                # Just need to check if the furthest position is in the partner oligo
                min_overlap_pos = (r1_3prime_pos + (min_overlap - 1)) % self.genome_length
                r1_3prime_overlap_sufficient = positions2_array[min_overlap_pos]
                
                # Combine condition with sufficient overlap check
                if positions2_array[r1_3prime_pos] and r1_3prime_overlap_sufficient and r2_direction and r1_3prime_pos != r2_start:
                    if verbose:
                        print(f"  Extending r1 (CCW) 3' end at {r1_3prime_pos} to r2's 5' end at {r2_start}")
                    
                    # 1. Simply extend to partner's 5' end (r2_start)
                    candidate_r1_start = r2_start
                    
                    # 2. Check if the length increases after extension
                    original_length = calc_length(r1_start, r1_end)
                    new_length = calc_length(candidate_r1_start, r1_end)
                    
                    if new_length > original_length:
                        # Length increased, proceed with extension
                        new_r1_start = candidate_r1_start
                    else:
                        # 3. Length decreased or remained same, set 3' end to one more than the strand's own 5' end
                        new_r1_start = (r1_end + 1) % self.genome_length
                    
                    # Only set flags if an actual extension occurred
                    if new_r1_start != r1_start:
                        extensions_made = True
                        r1_extended = True
                                
            # Check and extend the 3' end of the second oligo
            r2_extended = False
            if r2_direction:  # Clockwise -> 3' end is at r2_end
                r2_3prime_pos = r2_end
                # Check if there's sufficient overlap (min_overlap) at the 3' end
                # Just need to check if the furthest position is in the partner oligo
                min_overlap_pos = (r2_3prime_pos - (min_overlap - 1)) % self.genome_length
                r2_3prime_overlap_sufficient = positions1_array[min_overlap_pos]
                
                # Combine condition with sufficient overlap check
                if positions1_array[r2_3prime_pos] and r2_3prime_overlap_sufficient and not r1_direction and r2_3prime_pos != r1_end:
                    if verbose:
                        print(f"  Extending r2 (CW) 3' end at {r2_3prime_pos} to r1's 5' end at {r1_end}")
                    
                    # 1. Simply extend to partner's 5' end (r1_end)
                    candidate_r2_end = r1_end
                    
                    # 2. Check if the length increases after extension
                    original_length = calc_length(r2_start, r2_end)
                    new_length = calc_length(r2_start, candidate_r2_end)
                    
                    if new_length > original_length:
                        # Length increased, proceed with extension
                        new_r2_end = candidate_r2_end
                    else:
                        # 3. Length decreased or remained same, set 3' end to one less than the strand's own 5' end
                        new_r2_end = (r2_start - 1) % self.genome_length
                    
                    # Only set flags if an actual extension occurred
                    if new_r2_end != r2_end:
                        extensions_made = True
                        r2_extended = True
            else:  # Counterclockwise -> 3' end is at r2_start
                r2_3prime_pos = r2_start
                # Check if there's sufficient overlap (min_overlap) at the 3' end
                # Just need to check if the furthest position is in the partner oligo
                min_overlap_pos = (r2_3prime_pos + (min_overlap - 1)) % self.genome_length
                r2_3prime_overlap_sufficient = positions1_array[min_overlap_pos]
                
                # Combine condition with sufficient overlap check
                if positions1_array[r2_3prime_pos] and r2_3prime_overlap_sufficient and r1_direction and r2_3prime_pos != r1_start:
                    if verbose:
                        print(f"  Extending r2 (CCW) 3' end at {r2_3prime_pos} to r1's 5' end at {r1_start}")
                    
                    # 1. Simply extend to partner's 5' end (r1_start)
                    candidate_r2_start = r1_start
                    
                    # 2. Check if the length increases after extension
                    original_length = calc_length(r2_start, r2_end)
                    new_length = calc_length(candidate_r2_start, r2_end)
                    
                    if new_length > original_length:
                        # Length increased, proceed with extension
                        new_r2_start = candidate_r2_start
                    else:
                        # 3. Length decreased or remained same, set 3' end to one more than the strand's own 5' end
                        new_r2_start = (r2_end + 1) % self.genome_length
                    
                    # Only set flags if an actual extension occurred
                    if new_r2_start != r2_start:
                        extensions_made = True
                        r2_extended = True
            
            if extensions_made:
                # Check if extended oligos now cover the mutation site
                new_positions1_array = fill_positions(self._extension_bit_array1, new_r1_start, new_r1_end)
                new_positions2_array = fill_positions(self._extension_bit_array2, new_r2_start, new_r2_end)
                
                # Determine if mutation is propagated
                # For r1: check if (other oligo is mutant) AND (r1 now covers mutation site) AND (r1 didn't cover it before)
                r1_gets_mutation = (r2_mutation > 0) and new_positions1_array[mutation_site] and not r1_covers_mutation and r1_extended
                
                # For r2: check if (other oligo is mutant) AND (r2 now covers mutation site) AND (r2 didn't cover it before)
                r2_gets_mutation = (r1_mutation > 0) and new_positions2_array[mutation_site] and not r2_covers_mutation and r2_extended
                
                # Apply mutations if conditions are met
                new_r1_mutation = 1 if r1_gets_mutation else r1_mutation
                new_r2_mutation = 1 if r2_gets_mutation else r2_mutation
                
                if verbose:
                    if r1_gets_mutation:
                        print(f"  Reactant 1 acquired mutation during extension (now covers site {mutation_site})")
                    if r2_gets_mutation:
                        print(f"  Reactant 2 acquired mutation during extension (now covers site {mutation_site})")
                    
                    # Create updated reactants (without reactant indices for now)
                    new_r1 = (new_r1_start, new_r1_end, r1_direction, False, new_r1_mutation)
                    new_r2 = (new_r2_start, new_r2_end, r2_direction, False, new_r2_mutation)
                    
                    print(f"  Extended reactant 1: {reactant1[:5]} -> {new_r1}")
                    print(f"  Extended reactant 2: {reactant2[:5]} -> {new_r2}")
                else:
                    # Create updated reactants without printing
                    new_r1 = (new_r1_start, new_r1_end, r1_direction, False, new_r1_mutation)
                    new_r2 = (new_r2_start, new_r2_end, r2_direction, False, new_r2_mutation)
                
                # Add the new extended reactants with DUPLEX_COMPONENT_MARKER concentration (-1)
                # instead of 0.0 to ensure they're preserved during cleanup
                new_r1_idx = self.add_oligomer_type(new_r1[0], new_r1[1], new_r1[2], 
                                                self.DUPLEX_COMPONENT_MARKER, new_r1[3], new_r1[4])
                new_r2_idx = self.add_oligomer_type(new_r2[0], new_r2[1], new_r2[2], 
                                                self.DUPLEX_COMPONENT_MARKER, new_r2[3], new_r2[4])
                
                # Instead of updating the existing duplex, create a new one with the extended reactants
                # 1. Add the new extended duplex
                new_duplex_tuple = (dup_start, dup_end, True, True, dup_mutation, new_r1_idx, new_r2_idx)
                new_duplex_idx = self.add_oligomer_type(
                    dup_start, dup_end, True,  # Preserve original coordinates but ensure clockwise
                    conc,                       # Transfer the concentration to the new duplex
                    True, dup_mutation,         # Still a duplex with same mutation
                    new_r1_idx, new_r2_idx      # Point to the extended reactants
                )
                
                # 2. Set concentration of the original duplex to 0
                self.concentrations[duplex_idx] = 0
                
                if verbose:
                    print(f"  Created new extended duplex at index {new_duplex_idx} with reactants {new_r1_idx}, {new_r2_idx}")
                    print(f"  Set concentration of original duplex at index {duplex_idx} to 0")
                
                # Add to the extended oligos dictionary
                extended_oligos[self.current_oligomers[new_duplex_idx]] = conc
            else:
                if verbose:
                    print("  No extension needed (3' ends not within partner's span or already at correct positions)")
                # Add unextended duplex to the result
                extended_oligos[duplex_tuple] = conc
        
        return extended_oligos

    def extension_oligos_with_stalling(self, time=100.0, verbose=False, min_overlap=3, nonstalling_fraction=0.5):
        """
        Simulate extension of oligomers during PCR with strand-specific stalling behavior.
        
        When stalling mode is enabled, extension is inhibited when there's a mismatch
        at the mutation site between the extending oligomer and its partner.
        Each strand's extension is independent - if strand 1 has a mismatch, only
        strand 1's extension is subject to stalling, while strand 2 can extend normally.
        
        Args:
            time: Time for extension (not used in current implementation)
            verbose: Whether to print detailed output
            min_overlap: Minimum overlap required for annealing
            nonstalling_fraction: Fraction of concentration that extends under stalling conditions (0.0-1.0)
                
        Returns:
            Dictionary of extended oligomers and their concentrations
        """
        # Get all duplexes regardless of concentration
        duplexes = self.get_duplexes(0)
        if not duplexes:
            print("No duplexes found for extension")
            return {}
        
        print(f"Simulating extension with strand-specific stalling for {len(duplexes)} duplex types...")
        print(f"Non-stalling fraction: {nonstalling_fraction:.2f} (fraction that extends under stalling conditions)")
        
        # Mutation site (midpoint of genome)
        mutation_site = self.genome_length // 2
        
        # Create a dictionary for extended duplexes (to return)
        extended_oligos = {}
        
        # Initialize bit arrays for position tracking (reuse instead of recreating)
        if not hasattr(self, '_extension_bit_array1'):
            self._extension_bit_array1 = np.zeros(self.genome_length, dtype=bool)
            self._extension_bit_array2 = np.zeros(self.genome_length, dtype=bool)
        
        # Helper function to calculate length considering circular genome
        def calc_length(start, end):
            if end >= start:
                return end - start + 1
            else:
                return (self.genome_length - start) + end + 1
        
        # Optimized position checking using bit arrays
        def fill_positions(bit_array, start, end):
            bit_array.fill(False)  # Reset array
            if end >= start:
                bit_array[start:end+1] = True
            else:  # Wraps around
                bit_array[start:] = True
                bit_array[:end+1] = True
            return bit_array
        
        # Helper function to check if position is at mutation site
        def is_at_mutation_site(pos):
            return pos == mutation_site
        
        # Helper function to check for stalling condition
        def check_stalling_condition(extending_oligo_mutation, partner_oligo_mutation, three_prime_pos):
            """
            Check if stalling should occur.
            Stalling happens when:
            1. The 3' end is at the mutation site
            2. There's a mismatch: wt extending on mutant OR mutant extending on wt
            """
            if not is_at_mutation_site(three_prime_pos):
                return False
            
            # Check for mismatch: wt (0) extending on mutant (>0) or mutant (>0) extending on wt (0)
            is_mismatch = ((extending_oligo_mutation == 0 and partner_oligo_mutation > 0) or 
                          (extending_oligo_mutation > 0 and partner_oligo_mutation == 0))
            
            return is_mismatch
        
        # Track stalling statistics
        stalled_extensions = 0
        normal_extensions = 0
        partial_extensions = 0
        
        # Store all extension outcomes to apply at the end
        original_duplex_updates = {}  # Maps duplex_idx -> new_concentration
        new_duplex_accumulator = {}  # Maps duplex_tuple -> accumulated_concentration
        
        # Process each duplex
        for i, (duplex_idx, duplex_tuple, conc) in enumerate(duplexes):
            # Unpack components
            dup_start, dup_end, dup_direction, _, dup_mutation = duplex_tuple[:5]
            
            # Get reactant indices directly from the duplex tuple
            if len(duplex_tuple) >= 7:
                reactant1_idx, reactant2_idx = duplex_tuple[5], duplex_tuple[6]
            else:
                if verbose:
                    print(f"  WARNING: Duplex missing reactant indices - this shouldn't happen")
                continue
            
            # Check if these indices are valid
            if (reactant1_idx is None or reactant1_idx >= len(self.current_oligomers) or
                reactant2_idx is None or reactant2_idx >= len(self.current_oligomers)):
                if verbose:
                    print(f"  WARNING: Invalid reactant indices {reactant1_idx}, {reactant2_idx}")
                continue
                
            # Get the reactant tuples
            reactant1 = self.current_oligomers[reactant1_idx]
            reactant2 = self.current_oligomers[reactant2_idx]
            
            r1_start, r1_end, r1_direction, _, r1_mutation = reactant1[:5]
            r2_start, r2_end, r2_direction, _, r2_mutation = reactant2[:5]
            
            if verbose:
                print(f"\nExtension for duplex {i+1}: {duplex_tuple}, conc: {conc:.2f}")
            
            # Fill bit arrays with positions (reuse arrays instead of creating sets)
            positions1_array = fill_positions(self._extension_bit_array1, r1_start, r1_end)
            positions2_array = fill_positions(self._extension_bit_array2, r2_start, r2_end)
            
            # Track extension attempts and stalling conditions for each strand
            r1_extension_attempt = False
            r2_extension_attempt = False
            r1_stalling = False
            r2_stalling = False
            
            # Track if any extensions are possible and calculate new boundaries
            r1_can_extend = False
            r2_can_extend = False
            new_r1_start, new_r1_end = r1_start, r1_end
            new_r2_start, new_r2_end = r2_start, r2_end
            
            # Check if the oligos cover the mutation site before extension
            r1_covers_mutation = positions1_array[mutation_site]
            r2_covers_mutation = positions2_array[mutation_site]
            
            # Check and analyze the 3' end of the first oligo
            if r1_direction:  # Clockwise -> 3' end is at r1_end
                r1_3prime_pos = r1_end
                # Check if there's sufficient overlap (min_overlap) at the 3' end
                min_overlap_pos = (r1_3prime_pos - (min_overlap - 1)) % self.genome_length
                r1_3prime_overlap_sufficient = positions2_array[min_overlap_pos]
                
                if positions2_array[r1_3prime_pos] and r1_3prime_overlap_sufficient and not r2_direction and r1_3prime_pos != r2_end:
                    r1_extension_attempt = True
                    r1_stalling = check_stalling_condition(r1_mutation, r2_mutation, r1_3prime_pos)
                    
                    # Calculate new end position
                    candidate_r1_end = r2_end
                    original_length = calc_length(r1_start, r1_end)
                    new_length = calc_length(r1_start, candidate_r1_end)
                    
                    if new_length > original_length:
                        new_r1_end = candidate_r1_end
                        r1_can_extend = (new_r1_end != r1_end)
                    else:
                        new_r1_end = (r1_start - 1) % self.genome_length
                        r1_can_extend = (new_r1_end != r1_end)
            else:  # Counterclockwise -> 3' end is at r1_start
                r1_3prime_pos = r1_start
                min_overlap_pos = (r1_3prime_pos + (min_overlap - 1)) % self.genome_length
                r1_3prime_overlap_sufficient = positions2_array[min_overlap_pos]
                
                if positions2_array[r1_3prime_pos] and r1_3prime_overlap_sufficient and r2_direction and r1_3prime_pos != r2_start:
                    r1_extension_attempt = True
                    r1_stalling = check_stalling_condition(r1_mutation, r2_mutation, r1_3prime_pos)
                    
                    # Calculate new start position
                    candidate_r1_start = r2_start
                    original_length = calc_length(r1_start, r1_end)
                    new_length = calc_length(candidate_r1_start, r1_end)
                    
                    if new_length > original_length:
                        new_r1_start = candidate_r1_start
                        r1_can_extend = (new_r1_start != r1_start)
                    else:
                        new_r1_start = (r1_end + 1) % self.genome_length
                        r1_can_extend = (new_r1_start != r1_start)
                                
            # Check and analyze the 3' end of the second oligo
            if r2_direction:  # Clockwise -> 3' end is at r2_end
                r2_3prime_pos = r2_end
                min_overlap_pos = (r2_3prime_pos - (min_overlap - 1)) % self.genome_length
                r2_3prime_overlap_sufficient = positions1_array[min_overlap_pos]
                
                if positions1_array[r2_3prime_pos] and r2_3prime_overlap_sufficient and not r1_direction and r2_3prime_pos != r1_end:
                    r2_extension_attempt = True
                    r2_stalling = check_stalling_condition(r2_mutation, r1_mutation, r2_3prime_pos)
                    
                    # Calculate new end position
                    candidate_r2_end = r1_end
                    original_length = calc_length(r2_start, r2_end)
                    new_length = calc_length(r2_start, candidate_r2_end)
                    
                    if new_length > original_length:
                        new_r2_end = candidate_r2_end
                        r2_can_extend = (new_r2_end != r2_end)
                    else:
                        new_r2_end = (r2_start - 1) % self.genome_length
                        r2_can_extend = (new_r2_end != r2_end)
            else:  # Counterclockwise -> 3' end is at r2_start
                r2_3prime_pos = r2_start
                min_overlap_pos = (r2_3prime_pos + (min_overlap - 1)) % self.genome_length
                r2_3prime_overlap_sufficient = positions1_array[min_overlap_pos]
                
                if positions1_array[r2_3prime_pos] and r2_3prime_overlap_sufficient and r1_direction and r2_3prime_pos != r1_start:
                    r2_extension_attempt = True
                    r2_stalling = check_stalling_condition(r2_mutation, r1_mutation, r2_3prime_pos)
                    
                    # Calculate new start position
                    candidate_r2_start = r1_start
                    original_length = calc_length(r2_start, r2_end)
                    new_length = calc_length(candidate_r2_start, r2_end)
                    
                    if new_length > original_length:
                        new_r2_start = candidate_r2_start
                        r2_can_extend = (new_r2_start != r2_start)
                    else:
                        new_r2_start = (r2_end + 1) % self.genome_length
                        r2_can_extend = (new_r2_start != r2_start)
            
            # Calculate extension probabilities for each strand
            r1_extension_prob = nonstalling_fraction if (r1_extension_attempt and r1_stalling) else 1.0
            r2_extension_prob = nonstalling_fraction if (r2_extension_attempt and r2_stalling) else 1.0
            
            if verbose and (r1_stalling or r2_stalling):
                print(f"  Stalling detected:")
                if r1_stalling:
                    print(f"    r1 mutation mismatch at position {r1_3prime_pos}: r1={r1_mutation}, r2={r2_mutation}")
                if r2_stalling:
                    print(f"    r2 mutation mismatch at position {r2_3prime_pos}: r2={r2_mutation}, r1={r1_mutation}")
                print(f"    Extension probabilities: r1={r1_extension_prob:.2f}, r2={r2_extension_prob:.2f}")
            
            # Only proceed if at least one strand can extend
            if not (r1_can_extend or r2_can_extend):
                if verbose:
                    print("  No extension needed (3' ends not within partner's span or already at correct positions)")
                # Store that this duplex keeps its original concentration
                original_duplex_updates[duplex_idx] = conc
                extended_oligos[duplex_tuple] = conc
                continue
            
            # Calculate concentrations for different extension outcomes
            # Outcome probabilities (independent events):
            prob_neither_extends = (1 - r1_extension_prob) * (1 - r2_extension_prob) if (r1_can_extend and r2_can_extend) else (1 - (r1_extension_prob if r1_can_extend else r2_extension_prob))
            prob_r1_only_extends = r1_extension_prob * (1 - r2_extension_prob) if (r1_can_extend and r2_can_extend) else (r1_extension_prob if r1_can_extend and not r2_can_extend else 0)
            prob_r2_only_extends = (1 - r1_extension_prob) * r2_extension_prob if (r1_can_extend and r2_can_extend) else (r2_extension_prob if r2_can_extend and not r1_can_extend else 0)
            prob_both_extend = r1_extension_prob * r2_extension_prob if (r1_can_extend and r2_can_extend) else (r1_extension_prob if r1_can_extend and not r2_can_extend else r2_extension_prob if r2_can_extend and not r1_can_extend else 0)
            
            # Adjust probabilities if only one strand can extend
            if r1_can_extend and not r2_can_extend:
                prob_neither_extends = 1 - r1_extension_prob
                prob_r1_only_extends = r1_extension_prob
                prob_r2_only_extends = 0
                prob_both_extend = 0
            elif r2_can_extend and not r1_can_extend:
                prob_neither_extends = 1 - r2_extension_prob
                prob_r1_only_extends = 0
                prob_r2_only_extends = r2_extension_prob
                prob_both_extend = 0
            
            # Calculate actual concentrations
            conc_neither = conc * prob_neither_extends
            conc_r1_only = conc * prob_r1_only_extends
            conc_r2_only = conc * prob_r2_only_extends
            conc_both = conc * prob_both_extend
            
            if verbose:
                print(f"  Extension outcome concentrations:")
                print(f"    Neither extends: {conc_neither:.6f}")
                print(f"    Only r1 extends: {conc_r1_only:.6f}")
                print(f"    Only r2 extends: {conc_r2_only:.6f}")
                print(f"    Both extend: {conc_both:.6f}")
            
            # Store the concentration for the original duplex (unextended portion)
            original_duplex_updates[duplex_idx] = conc_neither
            if conc_neither > 0:
                extended_oligos[duplex_tuple] = conc_neither
            
            # Helper function to create extended reactants and handle mutation propagation
            def create_extended_reactants(extend_r1, extend_r2):
                # Determine new coordinates
                final_r1_start = new_r1_start if extend_r1 else r1_start
                final_r1_end = new_r1_end if extend_r1 else r1_end
                final_r2_start = new_r2_start if extend_r2 else r2_start
                final_r2_end = new_r2_end if extend_r2 else r2_end
                
                # Check mutation propagation only for extended strands
                final_r1_mutation = r1_mutation
                final_r2_mutation = r2_mutation
                
                if extend_r1:
                    # Check if extended r1 now covers the mutation site
                    new_positions1_array = fill_positions(self._extension_bit_array1, final_r1_start, final_r1_end)
                    r1_gets_mutation = (r2_mutation > 0) and new_positions1_array[mutation_site] and not r1_covers_mutation
                    if r1_gets_mutation:
                        final_r1_mutation = 1
                
                if extend_r2:
                    # Check if extended r2 now covers the mutation site
                    new_positions2_array = fill_positions(self._extension_bit_array2, final_r2_start, final_r2_end)
                    r2_gets_mutation = (r1_mutation > 0) and new_positions2_array[mutation_site] and not r2_covers_mutation
                    if r2_gets_mutation:
                        final_r2_mutation = 1
                
                # Create extended reactant oligomers (temporarily - we'll get their indices when we create the duplex)
                new_r1 = (final_r1_start, final_r1_end, r1_direction, False, final_r1_mutation)
                new_r2 = (final_r2_start, final_r2_end, r2_direction, False, final_r2_mutation)
                
                return new_r1, new_r2
            
            # Create extended duplex tuples for each outcome and accumulate concentrations
            # 2. Only r1 extends
            if conc_r1_only > 0:
                new_r1, new_r2 = create_extended_reactants(True, False)
                
                # We need to create the reactants first to get their indices
                new_r1_idx = self.add_oligomer_type(new_r1[0], new_r1[1], new_r1[2], 
                                                self.DUPLEX_COMPONENT_MARKER, new_r1[3], new_r1[4])
                new_r2_idx = self.add_oligomer_type(new_r2[0], new_r2[1], new_r2[2], 
                                                self.DUPLEX_COMPONENT_MARKER, new_r2[3], new_r2[4])
                
                # Create the duplex tuple
                r1_only_duplex_tuple = (dup_start, dup_end, True, True, dup_mutation, new_r1_idx, new_r2_idx)
                
                # Accumulate concentration for this duplex signature
                if r1_only_duplex_tuple in new_duplex_accumulator:
                    new_duplex_accumulator[r1_only_duplex_tuple] += conc_r1_only
                else:
                    new_duplex_accumulator[r1_only_duplex_tuple] = conc_r1_only
                
                if verbose:
                    print(f"    Will create r1-only extended duplex: r1 {reactant1[:5]} -> {new_r1}, r2 unchanged")
            
            # 3. Only r2 extends  
            if conc_r2_only > 0:
                new_r1, new_r2 = create_extended_reactants(False, True)
                
                new_r1_idx = self.add_oligomer_type(new_r1[0], new_r1[1], new_r1[2], 
                                                self.DUPLEX_COMPONENT_MARKER, new_r1[3], new_r1[4])
                new_r2_idx = self.add_oligomer_type(new_r2[0], new_r2[1], new_r2[2], 
                                                self.DUPLEX_COMPONENT_MARKER, new_r2[3], new_r2[4])
                
                r2_only_duplex_tuple = (dup_start, dup_end, True, True, dup_mutation, new_r1_idx, new_r2_idx)
                
                if r2_only_duplex_tuple in new_duplex_accumulator:
                    new_duplex_accumulator[r2_only_duplex_tuple] += conc_r2_only
                else:
                    new_duplex_accumulator[r2_only_duplex_tuple] = conc_r2_only
                
                if verbose:
                    print(f"    Will create r2-only extended duplex: r1 unchanged, r2 {reactant2[:5]} -> {new_r2}")
            
            # 4. Both strands extend
            if conc_both > 0:
                new_r1, new_r2 = create_extended_reactants(True, True)
                
                new_r1_idx = self.add_oligomer_type(new_r1[0], new_r1[1], new_r1[2], 
                                                self.DUPLEX_COMPONENT_MARKER, new_r1[3], new_r1[4])
                new_r2_idx = self.add_oligomer_type(new_r2[0], new_r2[1], new_r2[2], 
                                                self.DUPLEX_COMPONENT_MARKER, new_r2[3], new_r2[4])
                
                both_duplex_tuple = (dup_start, dup_end, True, True, dup_mutation, new_r1_idx, new_r2_idx)
                
                if both_duplex_tuple in new_duplex_accumulator:
                    new_duplex_accumulator[both_duplex_tuple] += conc_both
                else:
                    new_duplex_accumulator[both_duplex_tuple] = conc_both
                
                if verbose:
                    print(f"    Will create fully extended duplex: r1 {reactant1[:5]} -> {new_r1}, r2 {reactant2[:5]} -> {new_r2}")
            
            # Update statistics
            has_stalling = (r1_extension_attempt and r1_stalling) or (r2_extension_attempt and r2_stalling)
            has_partial = conc_r1_only > 0 or conc_r2_only > 0
            
            if has_stalling:
                stalled_extensions += 1
            if has_partial:
                partial_extensions += 1
            if conc_both > 0 and not has_stalling:
                normal_extensions += 1
        
        # Now apply all concentration updates atomically
        if verbose:
            print(f"\nApplying concentration updates...")
            print(f"  Updating {len(original_duplex_updates)} original duplexes")
            print(f"  Creating {len(new_duplex_accumulator)} new extended duplexes")
        
        # Update original duplex concentrations
        for duplex_idx, new_conc in original_duplex_updates.items():
            self.concentrations[duplex_idx] = new_conc
        
        # Create new extended duplexes with accumulated concentrations
        for duplex_tuple, total_conc in new_duplex_accumulator.items():
            # The duplex tuple already contains the correct reactant indices
            new_duplex_idx = self.add_oligomer_type(
                duplex_tuple[0], duplex_tuple[1], duplex_tuple[2],  # start, end, direction
                total_conc,                                        # accumulated concentration
                duplex_tuple[3], duplex_tuple[4],                  # is_duplex, mutation
                duplex_tuple[5], duplex_tuple[6]                   # reactant indices
            )
            
            # Add to the return dictionary
            extended_oligos[self.current_oligomers[new_duplex_idx]] = total_conc
            
            if verbose:
                print(f"    Created extended duplex with total concentration {total_conc:.6f}")
        
        print(f"Extension with strand-specific stalling complete:")
        print(f"  Normal extensions: {normal_extensions}")
        print(f"  Stalled extensions: {stalled_extensions}")
        print(f"  Partial extensions: {partial_extensions}")
        print(f"  Total duplexes processed: {len(duplexes)}")
        
        return extended_oligos

    def get_oligomers_above_threshold(self, threshold=0):
        """
        Return all oligomers with concentration above or equal to threshold.
        
        Args:
            threshold: Minimum concentration to consider (default 0 to include all oligomers)
        
        Returns:
            List of (oligomer_tuple, concentration) tuples sorted by concentration
        """
        if self.concentrations is None:
            return []
        
        result = []
        for i, oligo_type in enumerate(self.current_oligomers):
            if i < len(self.concentrations) and self.concentrations[i] >= threshold:
                result.append((oligo_type, self.concentrations[i]))
        
        # Sort by concentration (highest first)
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def get_single_stranded_oligos(self, threshold=0):
        """
        Get oligomers that are single-stranded (not duplexes).
        
        Args:
            threshold: Minimum concentration to consider (default 0 to include all oligos)
        
        Returns:
            List of (oligomer_tuple, concentration) tuples sorted by concentration
        """
        # Use the is_duplex field (index 3) to identify single-stranded oligos
        result = []
        for i, oligo_type in enumerate(self.current_oligomers):
            # Check that it's not a duplex and has sufficient concentration
            if (not oligo_type[3] and i < len(self.concentrations) and 
                self.concentrations[i] >= threshold and 
                self.concentrations[i] != self.DUPLEX_COMPONENT_MARKER):
                
                result.append((oligo_type, self.concentrations[i]))
        
        # Sort by concentration (highest first)
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def get_duplexes(self, threshold=0):
        """
        Identify all duplexes (annealed pairs) in the system.
        Returns actual indices along with duplex information for direct access.
        
        Args:
            threshold: Minimum concentration to consider (default 0 to include all duplexes)
        
        Returns:
            List of tuples (duplex_idx, duplex_tuple, concentration)
            where duplex_idx is the actual index in current_oligomers/concentrations
        """
        duplexes = []
        
        # Find oligomers that are marked as duplexes with concentration above threshold
        for idx, oligo_type in enumerate(self.current_oligomers):
            # Skip if the concentration is too low or index is out of bounds
            if idx >= len(self.concentrations) or self.concentrations[idx] < threshold:
                continue
            
            # Check if this is a duplex
            if oligo_type[3]:  # is_duplex flag is True
                # Get reactant indices from the extended tuple
                if len(oligo_type) >= 7:  # Ensure the tuple has reactant indices
                    reactant1_idx, reactant2_idx = oligo_type[5], oligo_type[6]
                    
                    # Verify these indices are valid
                    if (reactant1_idx is not None and reactant1_idx < len(self.current_oligomers) and
                        reactant2_idx is not None and reactant2_idx < len(self.current_oligomers)):
                        
                        # Add to duplexes list with the actual index for direct access:
                        # (duplex_idx, duplex_tuple, concentration)
                        duplexes.append((
                            idx,
                            oligo_type,
                            self.concentrations[idx]
                        ))
                        continue
                
                # If we didn't find valid reactant indices or couldn't add the duplex above,
                # add a warning and use placeholder reactants
                print(f"Warning: Duplex at index {idx} has missing or invalid reactant indices")
                
                # Add the duplex without reactant information
                duplexes.append((idx, oligo_type, self.concentrations[idx]))
        
        # Sort by concentration
        duplexes.sort(key=lambda x: x[2], reverse=True)
        
                
        return duplexes

    def simulate_melt_cycle(self, min_stable_length=60, verbose=False):
        """
        Simulate a melt cycle that selectively melts duplexes based on annealing length.
        Updates oligomer concentrations directly in the system.
        Uses actual indices to avoid lookup issues with extended duplexes.
        
        Args:
            min_stable_length: Minimum annealing length (in bp) for a duplex to remain stable
                            during the melt cycle. Duplexes with shorter annealing
                            regions will be melted.
            verbose: Whether to print detailed output
        
        Returns:
            Dictionary of melted products and their concentrations
        """
        if verbose:
            print("\nSimulating selective melt cycle...")
            print(f"Duplexes with annealing region < {min_stable_length} bp will melt")
        else:
            print("Simulating melt cycle...")
        
        # Get all current duplexes with their actual indices
        duplexes = self.get_duplexes(0)
        if not duplexes:
            if verbose:
                print("No duplexes found for melting")
            return {}
            
        if verbose:
            print(f"Found {len(duplexes)} duplexes to potentially melt")
        
        # Track statistics
        duplexes_melted = 0
        duplexes_stable = 0
        total_oligos_before = len([c for c in self.concentrations if c > 0])
        
        # Track melted products
        melted_products = {}
        
        for dup_idx, (duplex_idx, duplex_tuple, conc) in enumerate(duplexes):
            # Calculate the annealing region length
            # First, find the overlap between the two reactants
            d_start, d_end, d_dir, _, d_mutation = duplex_tuple[:5]
            
            # Get reactant indices directly from the duplex tuple
            if len(duplex_tuple) >= 7:
                reactant1_idx, reactant2_idx = duplex_tuple[5], duplex_tuple[6]
            else:
                # This shouldn't happen with the updated get_duplexes, but as a fallback:
                if verbose:
                    print(f"  WARNING: Duplex missing reactant indices - this shouldn't happen")
                continue
            
            # Check if these indices are valid
            if (reactant1_idx is None or reactant1_idx >= len(self.current_oligomers) or
                reactant2_idx is None or reactant2_idx >= len(self.current_oligomers)):
                if verbose:
                    print(f"  WARNING: Invalid reactant indices {reactant1_idx}, {reactant2_idx}")
                continue
                
            # Get the reactant tuples
            reactant1 = self.current_oligomers[reactant1_idx]
            reactant2 = self.current_oligomers[reactant2_idx]
            
            # Calculate annealing length using the centralized method
            annealing_length = self.calculate_annealing_length(reactant1, reactant2)
            
            if verbose:
                r1_start, r1_end, r1_dir, _, r1_mutation = reactant1[:5]
                r2_start, r2_end, r2_dir, _, r2_mutation = reactant2[:5]
                print(f"Duplex {duplex_idx} ({d_start}-{d_end}, {'CW' if d_dir else 'CCW'}, mutation={d_mutation}) has annealing length of {annealing_length} bp")
            
            # Check if this duplex should melt based on annealing length
            should_melt = annealing_length < min_stable_length
            
            if should_melt:
                # Add the concentration of the melted duplex to its reactants
                duplex_conc = self.concentrations[duplex_idx]
                
                # Check if reactant1_idx has DUPLEX_COMPONENT_MARKER concentration (-1)
                # If so, compensate for the marker
                if self.concentrations[reactant1_idx] == self.DUPLEX_COMPONENT_MARKER:
                    self.concentrations[reactant1_idx] = duplex_conc
                else:
                    self.concentrations[reactant1_idx] += duplex_conc
                
                # Same check for reactant2_idx
                if self.concentrations[reactant2_idx] == self.DUPLEX_COMPONENT_MARKER:
                    self.concentrations[reactant2_idx] = duplex_conc
                else:
                    self.concentrations[reactant2_idx] += duplex_conc
                
                # Zero out the duplex concentration
                self.concentrations[duplex_idx] = 0.0
                
                if verbose:
                    print(f"  MELTED: Distributed concentration {duplex_conc:.2f} to reactants {reactant1_idx} and {reactant2_idx}")
                duplexes_melted += 1
                
                # Add to melted products dictionary
                melted_products[self.current_oligomers[reactant1_idx]] = (self.concentrations[reactant1_idx], "Reactant 1")
                melted_products[self.current_oligomers[reactant2_idx]] = (self.concentrations[reactant2_idx], "Reactant 2")
            else:
                # This duplex remains stable - keep its concentration as is
                if verbose:
                    print(f"  STABLE: Duplex remains intact with concentration {conc:.2f}")
                duplexes_stable += 1
        
        # Track cycle statistics
        total_oligos_after = len([c for c in self.concentrations if c > 0])
        
        # Update cycle history if it exists
        if hasattr(self, 'cycle_history'):
            self.cycle_history['single_stranded'].append(len(self.get_single_stranded_oligos(0)))
            self.cycle_history['duplexes'].append(len(self.get_duplexes(0)))
            self.cycle_history['total_oligos'].append(total_oligos_after)
        
        if verbose:
            print(f"\nMelt cycle complete: {duplexes_melted} duplexes melted, {duplexes_stable} duplexes remained stable")
            print(f"Oligomers before: {total_oligos_before}, after: {total_oligos_after}")
        else:
            print(f"Melt cycle complete: {duplexes_melted} melted, {duplexes_stable} stable")
        
        return melted_products

    def cleanup_oligomers(self, cleanup_threshold=0):
        """
        Remove oligomers with concentration below threshold, but preserve:
        1. Oligomers with concentration >= cleanup_threshold
        2. Oligomers with the special DUPLEX_COMPONENT_MARKER concentration (-1)
        3. All participants in reaction pairs where:
        - The product duplex is kept, OR
        - Both reactants have non-zero concentrations
        
        Updates the system after cleanup and ensures analytical Jacobian is recalculated.
        
        Args:
            cleanup_threshold: Concentration threshold for removal
                
        Returns:
            Number of oligomers removed
        """
        print(f"Cleaning up oligomers with concentration < {cleanup_threshold}...")
        
        # Optimization 1: Vectorize Initial Filtering with NumPy operations
        keep_mask = (self.concentrations >= cleanup_threshold) | (self.concentrations == self.DUPLEX_COMPONENT_MARKER)
        concentration_keep = np.where(keep_mask)[0]
        
        # Use NumPy arrays for faster filtering
        reaction_pairs_array = np.array(self.reaction_pairs)
        if len(reaction_pairs_array) == 0:
            # Early return for empty reaction pairs
            keep_indices = sorted(concentration_keep)
            if len(keep_indices) == len(self.current_oligomers):
                print("No oligomers need to be removed")
                return 0
            
            # Keep only the selected oligomers
            new_types = [self.current_oligomers[i] for i in keep_indices]
            new_concentrations = self.concentrations[keep_indices]
            
            # Update the oligomer indices dictionary
            old_to_new_index = {old_idx: i for i, old_idx in enumerate(keep_indices)}
            new_indices = {oligo_type: old_to_new_index[old_idx] 
                         for oligo_type, old_idx in self.oligomer_indices.items() 
                         if old_idx in old_to_new_index}
            
            # Apply updates
            self.current_oligomers = new_types
            self.oligomer_indices = new_indices
            self.concentrations = new_concentrations
            self.reaction_pairs = []
            from scipy import sparse
            self.reaction_matrix = sparse.csr_matrix((len(new_types), 0))
            self.processed_oligomers = set()
            
            # Clear Jacobian - will be recalculated when needed
            if hasattr(self, '_jac_fn'):
                delattr(self, '_jac_fn')
            
            removed_count = len(self.current_oligomers) - len(keep_indices)
            print(f"Removed {removed_count} oligomers. System now has {len(self.current_oligomers)} oligomers and {len(self.reaction_pairs)} reactions")
            return removed_count
        
        # Optimization 1 & 2: Use NumPy for reaction pair filtering
        # Extract columns for faster access
        reactants1 = reaction_pairs_array[:, 0]
        reactants2 = reaction_pairs_array[:, 1]
        products = reaction_pairs_array[:, 2]
        
        # Find which products are being kept
        product_kept = np.isin(products, concentration_keep)
        
        # Find which reactions have both reactants with positive concentrations
        conc_len = len(self.concentrations)
        valid_indices = np.where(
            (reactants1 < conc_len) & (reactants2 < conc_len) &
            (self.concentrations[reactants1] > 0) & (self.concentrations[reactants2] > 0)
        )[0]
        
        # Combine the conditions
        keep_rxn_mask = product_kept | np.isin(np.arange(len(reaction_pairs_array)), valid_indices)
        keep_rxn_indices = np.where(keep_rxn_mask)[0]
        
        # Get all reaction participants that need to be kept
        all_participants = np.unique(np.concatenate([
            reactants1[keep_rxn_mask],
            reactants2[keep_rxn_mask],
            products[keep_rxn_mask]
        ]))
        
        # Optimization 2: Combine all indices efficiently using NumPy set operations
        keep_indices = np.unique(np.concatenate([concentration_keep, all_participants]))
        keep_indices = sorted(keep_indices)
        
        # Create a set version immediately for efficient membership testing later
        keep_indices_set = set(keep_indices)
        
        # If nothing to remove, return early
        if len(keep_indices) == len(self.current_oligomers):
            print("No oligomers need to be removed")
            return 0
        
        # Report what's being kept
        print(f'Keeping {len(keep_indices)} oligomers out of {len(self.current_oligomers)}')
        print(f'Keeping {len(keep_rxn_indices)} reaction pairs out of {len(self.reaction_pairs)}')
        
        # Optimization 3: Faster index remapping with NumPy arrays
        max_old_idx = max(keep_indices) if keep_indices else 0
        old_to_new_mapping = np.full(max_old_idx + 1, -1, dtype=np.int32)
        old_to_new_mapping[keep_indices] = np.arange(len(keep_indices), dtype=np.int32)
        
        # Create mapping from old indices to new indices (for dict access)
        old_to_new_index = {old_idx: int(old_to_new_mapping[old_idx]) for old_idx in keep_indices}
        
        # Create new lists for the kept oligomers (minimize allocations)
        new_types = [self.current_oligomers[i] for i in keep_indices]
        new_concentrations = self.concentrations[keep_indices]
        
        # Update the oligomer indices dictionary with pre-allocation
        new_indices = {}
        for oligo_type, old_idx in self.oligomer_indices.items():
            if old_idx in old_to_new_index:
                new_indices[oligo_type] = old_to_new_index[old_idx]
        
        # Update reaction pairs efficiently
        new_reaction_pairs = []
        # Pre-allocate list to avoid resizing
        new_reaction_pairs = [(0, 0, 0)] * len(keep_rxn_indices)
        for new_idx, old_idx in enumerate(keep_rxn_indices):
            i, j, k = reaction_pairs_array[old_idx]
            new_reaction_pairs[new_idx] = (
                old_to_new_mapping[i],
                old_to_new_mapping[j],
                old_to_new_mapping[k]
            )
        
        # Update references to reactants in duplex tuples
        for idx, oligo in enumerate(new_types):
            if oligo[3] and len(oligo) >= 7:  # If it's a duplex with reactant indices
                reactant1_idx, reactant2_idx = oligo[5], oligo[6]
                
                # Update reactant indices if they're being kept
                new_reactant1_idx = old_to_new_index.get(reactant1_idx) if reactant1_idx in keep_indices_set else None
                new_reactant2_idx = old_to_new_index.get(reactant2_idx) if reactant2_idx in keep_indices_set else None
                
                # Create updated oligo with new reactant indices
                updated_oligo = oligo[:5] + (new_reactant1_idx, new_reactant2_idx)
                
                # Replace the oligo in new_types
                new_types[idx] = updated_oligo
                
                # Update the indices dictionary
                if oligo in new_indices:
                    del new_indices[oligo]
                new_indices[updated_oligo] = idx
        
        # Optimization 4: Direct CSR matrix construction
        from scipy import sparse
        
        if new_reaction_pairs:
            # Get the maximum index and reaction count
            n_kept_oligomers = len(new_types)
            n_kept_reactions = len(new_reaction_pairs)
            
            # Pre-allocate arrays for CSR format
            # Each reaction has 3 non-zero entries (-1, -1, 1)
            nnz = n_kept_reactions * 3
            data = np.empty(nnz, dtype=np.float64)
            indices = np.empty(nnz, dtype=np.int32)
            
            # Create indptr array for CSR format
            # indptr[i] gives the starting index in 'indices' for row i
            indptr = np.zeros(n_kept_oligomers + 1, dtype=np.int32)
            
            # Count non-zeros per row first to build indptr
            row_counts = np.zeros(n_kept_oligomers, dtype=np.int32)
            for i, j, k in new_reaction_pairs:
                row_counts[i] += 1
                row_counts[j] += 1
                row_counts[k] += 1
            
            # Set up indptr using cumulative sum
            indptr[1:] = np.cumsum(row_counts)
            
            # Fill data and indices arrays
            row_positions = np.copy(indptr[:-1])  # Current position in each row
            for col, (i, j, k) in enumerate(new_reaction_pairs):
                # Reactant i
                pos = row_positions[i]
                data[pos] = -1
                indices[pos] = col
                row_positions[i] += 1
                
                # Reactant j
                pos = row_positions[j]
                data[pos] = -1
                indices[pos] = col
                row_positions[j] += 1
                
                # Product k
                pos = row_positions[k]
                data[pos] = 1
                indices[pos] = col
                row_positions[k] += 1
            
            # Create CSR matrix directly
            new_matrix = sparse.csr_matrix(
                (data, indices, indptr),
                shape=(n_kept_oligomers, n_kept_reactions)
            )
        else:
            # Empty sparse matrix
            new_matrix = sparse.csr_matrix((len(new_types), 0))
        
        # Update processed oligomers set
        new_processed = {old_to_new_index[i] for i in self.processed_oligomers 
                        if i in keep_indices_set}
        
        # Clean up Jacobian - will be recalculated when needed
        if hasattr(self, '_jac_fn'):
            delattr(self, '_jac_fn')
        
        # Apply the updates
        old_oligo_count = len(self.current_oligomers)
        self.current_oligomers = new_types
        self.oligomer_indices = new_indices
        self.concentrations = new_concentrations
        self.reaction_pairs = new_reaction_pairs
        self.reaction_matrix = new_matrix
        self.processed_oligomers = new_processed
        
        # Rebuild reaction array for simulation
        self._reaction_pairs_array = np.array(self.reaction_pairs)
        
        removed_count = old_oligo_count - len(self.current_oligomers)
        print(f"Removed {removed_count} oligomers. System now has {len(self.current_oligomers)} oligomers and {len(self.reaction_pairs)} reactions")
        
        return removed_count

    def reaction_rates(self, t, concentrations):
        """
        Calculate rates for all reactions at given concentrations.
        Vectorized implementation for better performance.
        
        Args:
            t: Time point (not used, required by solver)
            concentrations: Array of current concentrations
            
        Returns:
            Array of reaction rates
        """
        # No reactions case
        if len(self.reaction_pairs) == 0:
            return np.array([])
            
        # Use JIT-compiled version if available for better performance
        if NUMBA_AVAILABLE and hasattr(self, '_reaction_pairs_array'):
            return _numba_reaction_rates(concentrations, self._reaction_pairs_array)
        
        # Fallback to standard version
        # Extract reactant indices
        reactant1_indices = np.array([pair[0] for pair in self.reaction_pairs])
        reactant2_indices = np.array([pair[1] for pair in self.reaction_pairs])
        
        # Vectorized rate calculation: k * [A] * [B] for all reactions at once
        # Using a fixed rate constant k = 1.0
        k = 1.0
        
        # Compute all rates in a single vectorized operation
        rates = k * concentrations[reactant1_indices] * concentrations[reactant2_indices]
        
        return rates

    def concentration_derivatives(self, t, concentrations):
        """
        Calculate derivatives of all concentrations.
        This is the core of the ODE system.
        Optimized version for large systems.
        
        Args:
            t: Time point (not used, required by solver)
            concentrations: Array of current concentrations
            
        Returns:
            Array of concentration derivatives
        """
        
        # No reactions case - return zeros
        if len(self.reaction_pairs) == 0:
            return np.zeros_like(concentrations)
        
        # Use JIT-compiled version if available for better performance
        if (NUMBA_AVAILABLE and hasattr(self, '_reaction_pairs_array') and 
            hasattr(self, '_csr_matrix_data')):
                
            # Get rates using numba-optimized function
            rates = _numba_reaction_rates(concentrations, self._reaction_pairs_array)
            # Calculate derivatives using numba-optimized sparse matrix multiplication
            derivatives = _numba_sparse_matmul(
                rates, 
                self._csr_matrix_data, 
                self._csr_matrix_indices, 
                self._csr_matrix_indptr, 
                self._csr_matrix_shape
            )
                            
            return derivatives
            
        # Fallback to standard version
        # Calculate rates for all reactions (vectorized)
        rates = self.reaction_rates(t, concentrations)
        
        # Calculate derivatives using sparse matrix multiplication if available
        # Otherwise, use normal matrix multiplication
        try:
            from scipy import sparse
            # Convert to sparse format if the matrix is large enough to benefit
            if len(rates) > 1000 and not isinstance(self.reaction_matrix, sparse.spmatrix):
                # Cache the sparse version of the reaction matrix if not already done
                if not hasattr(self, '_sparse_reaction_matrix'):
                    self._sparse_reaction_matrix = sparse.csr_matrix(self.reaction_matrix)
                
                # Verify matrix shape matches concentrations length
                    
                derivatives = self._sparse_reaction_matrix @ rates
            else:
                # Verify matrix shape
                    
                derivatives = self.reaction_matrix @ rates
                                
        except ImportError:
            # Fall back to standard matrix multiplication if scipy is not available
            derivatives = self.reaction_matrix @ rates
        
        return derivatives

    def plot_annealing_trajectories(self, t, y, cycle_num=0, visualization_threshold=0):
        """
        Plot concentration trajectories during annealing.
        
        Args:
            t: Time points
            y: Concentration values over time
            cycle_num: PCR cycle number for plot labeling
            visualization_threshold: Minimum concentration to display trajectories
        """
        plt.figure(figsize=(12, 8))
        
        # Get indices of oligos with initial conc above visualization threshold
        mask = y[:, 0] > visualization_threshold
        
        # Plot each trajectory that starts above visualization threshold
        for i in range(len(self.current_oligomers)):
            if mask[i]:
                oligo = self.current_oligomers[i]
                start, end, is_cw, is_duplex,is_mut = oligo[:5]
                
                # Skip if this is a duplex
                if is_duplex:
                    continue
                    
                label = f"{start}-{end} {'CW' if is_cw else 'CCW'} {'mut' if is_mut else 'wt'}"
                plt.plot(t, y[i], label=label)
        
        plt.yscale('log')
        plt.ylim(1e-3, 1e2)  # Set y-axis limits
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.title(f'Single-Stranded Oligo Concentrations During Annealing (Cycle {cycle_num})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self._output_dir, f'annealing_trajectories_cycle{cycle_num}.png'))
        plt.close()

    def simulate(self, t_end=100.0, rtol=1e-8, atol=1e-9, verbose=False, cycle_num=0, cleanup_threshold=0.0, early_termination=True, stability_threshold=0.01):
        """
        Simulate the annealing process by integrating the ODEs using the BDF method.
        Uses analytical Jacobian for improved performance.
        
        Args:
            t_end: End time for simulation (default: 100.0)
            rtol: Relative tolerance (default: 1e-6)
            atol: Absolute tolerance (default: 1e-9)
            verbose: Whether to print detailed output and save plots
            cycle_num: Current PCR cycle number for plot labeling
            cleanup_threshold: Concentration threshold below which oligomers are considered negligible
            early_termination: Whether to terminate the ODE solver early when concentrations stabilize
            stability_threshold: Percentage change threshold for concentration stability (default: 0.01%)
            
        Returns:
            Solution object from solve_ivp
        """
        import time
        import numpy as np
        from scipy.integrate import solve_ivp
        
        start_time = time.time()
        
        if len(self.reaction_pairs) == 0:
            print("No reactions found. Skipping simulation.")
            return None
        
        # Prepare for simulation
        n_reactions = len(self.reaction_pairs)
        n_oligomers = len(self.concentrations)
        print(f"Simulating annealing with {n_reactions} reactions...")
        
        # Memory optimization for the reaction pairs array
        if NUMBA_AVAILABLE:
            if (not hasattr(self, '_reaction_pairs_array') or 
                self._reaction_pairs_array.shape[0] != n_reactions):
                self._reaction_pairs_array = np.array([[pair[0], pair[1]] for pair in self.reaction_pairs], 
                                                     dtype=np.int32)
            print("Using Numba JIT acceleration for ODE integration")
        
        # Set up sparse matrix for better performance
        try:
            from scipy import sparse
            
            # Create or update sparse matrix if needed
            if (not hasattr(self, '_sparse_reaction_matrix') or 
                not isinstance(self._sparse_reaction_matrix, sparse.spmatrix) or
                self._sparse_reaction_matrix.shape != self.reaction_matrix.shape):
                
                # Create sparse matrix (CSR format for efficient matrix-vector multiply)
                if isinstance(self.reaction_matrix, sparse.spmatrix):
                    self._sparse_reaction_matrix = self.reaction_matrix.tocsr()
                else:
                    self._sparse_reaction_matrix = sparse.csr_matrix(self.reaction_matrix)
            
            # Extract CSR components for Numba if available
            if NUMBA_AVAILABLE:
                if (not hasattr(self, '_csr_matrix_data') or 
                    len(self._csr_matrix_data) != len(self._sparse_reaction_matrix.data)):
                    self._csr_matrix_data = self._sparse_reaction_matrix.data
                    self._csr_matrix_indices = self._sparse_reaction_matrix.indices
                    self._csr_matrix_indptr = self._sparse_reaction_matrix.indptr
                    self._csr_matrix_shape = self._sparse_reaction_matrix.shape
            
        except ImportError:
            print("SciPy sparse matrix support not available")
        
        # Focus on time range with most activity
        # Dense sampling in 0-10 range, sparser afterwards
        t_eval = np.logspace(np.log10(1e-6 + cleanup_threshold), np.log10(t_end), 100)
        
        # Set up BDF solver with Jacobian optimization
        solver_kwargs = {
            'method': 'BDF',  # Best for stiff chemical kinetics
            'rtol': rtol,
            'atol': atol,
            't_eval': t_eval
        }
        
        # Set up analytical Jacobian for improved performance
        if NUMBA_AVAILABLE:
            # If not already set up, create the analytical Jacobian
            if not hasattr(self, '_jac_fn'):
                print("WARNING: Setting up analytical Jacobian within simulate... This seems wrong - AM. Remove it.")
                self._setup_analytical_jacobian()
            
            # Add the Jacobian function to solver kwargs
            if hasattr(self, '_jac_fn') and self._jac_fn is not None:
                solver_kwargs['jac'] = self._jac_fn
                print("Using analytical Jacobian for ODE integration")
        
        # Add early termination if requested
        if early_termination:
            # Use closure to maintain state between calls
            previous_y = None
            step_count = 0
            min_steps = 5  # Require at least this many steps before allowing termination
            
            def termination_event(t, y):
                nonlocal previous_y, step_count
                
                # Increment step counter
                step_count += 1
                
                # Don't terminate before minimum steps
                if step_count < min_steps:
                    return 1.0
                
                # Get current derivatives
                current_derivs = self.concentration_derivatives(t, y)
                
                # Store initial derivatives on first call after min_steps
                if step_count == min_steps:
                    self._initial_derivs = np.abs(current_derivs) + 1e-10  # Avoid division by zero
                
                # Calculate normalized derivative changes
                deriv_ratio = np.abs(current_derivs) / self._initial_derivs
                
                # Check if concentrations are very small (relative to 1.0)
                conc_is_small = y < cleanup_threshold
                
                # Check if derivatives are small compared to initial
                deriv_is_small = deriv_ratio < stability_threshold
                
                # Terminate if all values meet at least one condition
                if np.all(np.logical_or(conc_is_small, deriv_is_small)):
                    print(f"Early termination at t={t:.3f}, step {step_count}")
                    return 0.0
                
                return 1.0
            
            # Add to solver
            termination_event.terminal = True
            solver_kwargs['events'] = termination_event
        
        # Run the integration
        solve_start = time.time()
        print(f"Solve_ivp called with {len(self.concentrations)} concentrations")
                        
        print(f"Initial derivatives: {len(self.concentration_derivatives(0, self.concentrations))}")
        result = solve_ivp(
            self.concentration_derivatives,
            (0, t_end),
            self.concentrations,
            **solver_kwargs
        )
        solve_time = time.time() - solve_start


        # Update final concentrations
        n = min(len(self.concentrations), result.y.shape[0])
        self.concentrations[:n] = result.y[:n, -1]
        
        # Print simulation stats
        total_time = time.time() - start_time
        print(f"BDF simulation completed with {len(result.t)} time points in {total_time:.2f} seconds")
        if not result.success:
            print(f"Warning: Integration may not be accurate. Message: {result.message}")
        
        return result

    def simulate_RK45(self, t_end=100.0, rtol=1e-5, atol=1e-6, verbose=False, cycle_num=0, cleanup_threshold=0.0, early_termination=True, stability_threshold=0.01):
        """
        Simulate the annealing process by integrating the ODEs using the RK45 method.
        
        Args:
            t_end: End time for simulation (default: 100.0)
            rtol: Relative tolerance (default: 1e-8)
            atol: Absolute tolerance (default: 1e-9)
            verbose: Whether to print detailed output and save plots
            cycle_num: Current PCR cycle number for plot labeling
            cleanup_threshold: Concentration threshold below which oligomers are considered negligible
            early_termination: Whether to terminate the ODE solver early when concentrations stabilize
            stability_threshold: Percentage change threshold for concentration stability (default: 0.01%)
            
        Returns:
            Solution object from solve_ivp
        """
        import time
        import numpy as np
        from scipy.integrate import solve_ivp
        
        start_time = time.time()
        
        if len(self.reaction_pairs) == 0:
            print("No reactions found. Skipping simulation.")
            return None
        
        # Prepare for simulation
        n_reactions = len(self.reaction_pairs)
        n_oligomers = len(self.concentrations)
        print(f"Simulating annealing with {n_reactions} reactions using RK45...")
        
        # Memory optimization for the reaction pairs array
        if NUMBA_AVAILABLE:
            if (not hasattr(self, '_reaction_pairs_array') or 
                self._reaction_pairs_array.shape[0] != n_reactions):
                self._reaction_pairs_array = np.array([[pair[0], pair[1]] for pair in self.reaction_pairs], 
                                                     dtype=np.int32)
            print("Using Numba JIT acceleration for ODE integration")
        
        # Set up sparse matrix for better performance
        from scipy import sparse
        
        # Create or update sparse matrix if needed
        if (not hasattr(self, '_sparse_reaction_matrix') or 
            not isinstance(self._sparse_reaction_matrix, sparse.spmatrix) or
            self._sparse_reaction_matrix.shape != self.reaction_matrix.shape):
            
            # Create sparse matrix (CSR format for efficient matrix-vector multiply)
            if isinstance(self.reaction_matrix, sparse.spmatrix):
                self._sparse_reaction_matrix = self.reaction_matrix.tocsr()
            else:
                self._sparse_reaction_matrix = sparse.csr_matrix(self.reaction_matrix)
        
        # Extract CSR components for Numba if available
        if NUMBA_AVAILABLE:
            if (not hasattr(self, '_csr_matrix_data') or 
                len(self._csr_matrix_data) != len(self._sparse_reaction_matrix.data)):
                self._csr_matrix_data = self._sparse_reaction_matrix.data
                self._csr_matrix_indices = self._sparse_reaction_matrix.indices
                self._csr_matrix_indptr = self._sparse_reaction_matrix.indptr
                self._csr_matrix_shape = self._sparse_reaction_matrix.shape
        
        
        # Focus on time range with most activity
        # Dense sampling in 0-10 range, sparser afterwards
        t_eval = np.logspace(np.log10(1e-6 + cleanup_threshold), np.log10(t_end), 100)
        
        # Set up RK45 solver
        solver_kwargs = {
            'method': 'RK45',  # Explicit Runge-Kutta method of order 5(4)
            'rtol': rtol,
            'atol': atol,
            't_eval': t_eval
        }
        
        # Add early termination if requested
        if early_termination:
            # Use closure to maintain state between calls
            previous_y = None
            step_count = 0
            min_steps = 5  # Require at least this many steps before allowing termination
            
            def termination_event(t, y):
                nonlocal previous_y, step_count
                
                # Increment step counter
                step_count += 1
                
                # Don't terminate before minimum steps
                if step_count < min_steps:
                    return 1.0
                
                # Get current derivatives
                current_derivs = self.concentration_derivatives(t, y)
                
                # Store initial derivatives on first call after min_steps
                if step_count == min_steps:
                    self._initial_derivs = np.abs(current_derivs) + 1e-10  # Avoid division by zero
                
                # Calculate normalized derivative changes
                deriv_ratio = np.abs(current_derivs) / self._initial_derivs
                
                # Check if concentrations are very small (relative to 1.0)
                conc_is_small = y < cleanup_threshold
                
                # Check if derivatives are small compared to initial
                deriv_is_small = deriv_ratio < stability_threshold
                
                # Terminate if all values meet at least one condition
                if np.all(np.logical_or(conc_is_small, deriv_is_small)):
                    print(f"Early termination at t={t:.3f}, step {step_count}")
                    return 0.0
                
                return 1.0
            
            # Add to solver
            termination_event.terminal = True
            solver_kwargs['events'] = termination_event
        
        # Run the integration
        solve_start = time.time()
        print(f"Solve_ivp called with {len(self.concentrations)} concentrations")
                        
        print(f"Initial derivatives: {len(self.concentration_derivatives(0, self.concentrations))}")
        result = solve_ivp(
            self.concentration_derivatives,
            (0, t_end),
            self.concentrations,
            **solver_kwargs
        )
        solve_time = time.time() - solve_start
        
        
        # Update final concentrations
        n = min(len(self.concentrations), result.y.shape[0])
        self.concentrations[:n] = result.y[:n, -1]
        
        # Print simulation stats
        total_time = time.time() - start_time
        print(f"RK45 simulation completed with {len(result.t)} time points in {total_time:.2f} seconds")
        if not result.success:
            print(f"Warning: Integration may not be accurate. Message: {result.message}")
        
        return result

    def _setup_analytical_jacobian(self):
        """
        Set up analytical Jacobian function for improved performance.
        Uses Numba compilation for better performance.
        """
        if not NUMBA_AVAILABLE:
            return
        
        try:
            # Define the analytical Jacobian function
            from numba import njit
            from scipy import sparse
            import numpy as np
            
            # Get the shape from system dimensions
            n_oligomers = len(self.concentrations)
            
            # Create reaction pairs array if needed
            if not hasattr(self, '_reaction_pairs_array') or len(self._reaction_pairs_array) != len(self.reaction_pairs):
                self._reaction_pairs_array = np.array(self.reaction_pairs)
            
            @njit(fastmath=True)
            def _analytical_jac(t, y, reaction_pairs_array):
                """Compute the Jacobian matrix analytically"""
                n_oligomers = len(y)
                n_reactions = len(reaction_pairs_array)
                
                # Create dense matrix for the Jacobian
                jac = np.zeros((n_oligomers, n_oligomers), dtype=np.float32)
                
                # For each reaction, update the Jacobian elements
                for r in range(n_reactions):
                    i, j, k = reaction_pairs_array[r, 0], reaction_pairs_array[r, 1], reaction_pairs_array[r, 2]
                    
                    # Skip if indices are out of bounds
                    if i >= n_oligomers or j >= n_oligomers or k >= n_oligomers:
                        print(f'WARNING AM: analytical Jacobian: Skipping reaction {r} because indices are out of bounds: {i}, {j}, {k}')
                        continue
                    
                    # Compute partial derivatives
                    # d(rate)/d[i] = k * [j]
                    # d(rate)/d[j] = k * [i]
                    k_val = 1.0  # Fixed rate constant
                    d_rate_d_i = k_val * y[j]
                    d_rate_d_j = k_val * y[i]
                    
                    # Update Jacobian entries for this reaction
                    # d[i]/d[i] = -d(rate)/d[i]
                    jac[i, i] -= d_rate_d_i
                    # d[i]/d[j] = -d(rate)/d[j]
                    jac[i, j] -= d_rate_d_j
                    # d[j]/d[j] = -d(rate)/d[j]
                    jac[j, j] -= d_rate_d_j
                    # d[j]/d[i] = -d(rate)/d[i]
                    jac[j, i] -= d_rate_d_i
                    # d[k]/d[i] = d(rate)/d[i]
                    jac[k, i] += d_rate_d_i
                    # d[k]/d[j] = d(rate)/d[j]
                    jac[k, j] += d_rate_d_j
                
                return jac
            
            # Create a wrapper function to interface with solve_ivp
            def jac_wrapper(t, y):
                return _analytical_jac(t, y, self._reaction_pairs_array)
            
            # Store the function
            self._jac_fn = jac_wrapper
            
            print(f"Analytical Jacobian function set up for {n_oligomers} oligomers and {len(self.reaction_pairs)} reactions")
        except Exception as e:
            print(f"Failed to set up analytical Jacobian: {e}")
            import traceback
            traceback.print_exc()
    
    def _multi_stage_integration(self, t_end, solver_kwargs):
        """
        Implement multi-stage integration for very large systems.
        Breaks the integration into stages for better performance.
        
        Args:
            t_end: Final integration time
            solver_kwargs: Solver keyword arguments
            
        Returns:
            Solution object from solve_ivp
        """
        from scipy.integrate import solve_ivp
        import numpy as np
        import time
        
        # Use logarithmic time stages to focus on faster initial reactions
        # Most activity happens in the first 10% of time
        n_stages = 5
        stage_times = np.logspace(np.log10(0.01), np.log10(t_end), n_stages)
        
        print(f"Performing {n_stages}-stage integration: {stage_times}")
        
        # Initial conditions
        y0 = self.concentrations.copy()
        t_start = 0
        
        # Capture all time points and solutions
        all_t = [0]
        all_y = [y0]
        
        # Track solution success
        success = True
        message = ""
        n_fun_calls = 0
        n_jac_calls = 0
        
        for i, t_stage in enumerate(stage_times):
            print(f"Stage {i+1}/{n_stages}: t = {t_start:.2f} to {t_stage:.2f}")
            
            # Solve for this stage
            stage_result = solve_ivp(
                self.concentration_derivatives,
                (t_start, t_stage),
                y0,
                **solver_kwargs
            )
            
            # Update tracking
            success = success and stage_result.success
            if not stage_result.success:
                message = stage_result.message
            
            n_fun_calls += stage_result.nfev
            if hasattr(stage_result, 'njev'):
                n_jac_calls += stage_result.njev
            
            # Store results (skip first point to avoid duplicating t=0)
            if i > 0:
                all_t.extend(stage_result.t.tolist())
                all_y.extend(stage_result.y.T.tolist())
            else:
                all_t.extend(stage_result.t[1:].tolist())
                all_y.extend(stage_result.y.T[1:].tolist())
            
            # Update initial conditions for next stage
            y0 = stage_result.y[:, -1]
            t_start = t_stage
            
            # Print progress
            print(f"  Completed stage {i+1}: {len(stage_result.t)} points, {stage_result.nfev} function evals")
        
        # Convert lists to arrays
        t = np.array(all_t)
        y = np.array(all_y).T
        
        # Create a solution object with the results
        class Solution:
            def __init__(self, t, y, success, message, nfev, njev):
                self.t = t
                self.y = y
                self.success = success
                self.message = message
                self.nfev = nfev
                self.njev = njev
        
        return Solution(t, y, success, message, n_fun_calls, n_jac_calls)

    def run_pcr_protocol(self, cycles=10, min_stable_length=70, annealing_time=100.0, cleanup_threshold=0, 
                         verbose=False, save_plots=True, enable_nucleation=True, nucleation_oligo_length=6, 
                         K=1.0, random_seed=None, Pmax=1.0, 
                         nucleation_sparse_sampling=0.05, sort_by_concentration=False, min_overlap=3,
                         genome_viz_verbose='off', max_oligos=50, state_save_path=None, visualization_threshold=0,
                         stability_threshold=0.01, annealing_verbose=False, use_stalling=False, nonstalling_fraction=1.0,
                         output_dir='plots'):
        """
        Run a complete PCR protocol with multiple cycles.
        Updated for new oligomer data structure.
        
        This version saves the exact state after each melt cycle for accurate visualization.
        
        Args:
            cycles: Number of PCR cycles to run
            min_stable_length: Minimum stable length for duplexes
            annealing_time: Time for each annealing simulation
            cleanup_threshold: Concentration threshold for cleanup
            verbose: Whether to print detailed output
            save_plots: Whether to save plots
            enable_nucleation: Whether to enable de novo nucleation
            nucleation_oligo_length: Length of de novo oligos
            K: Concentration parameter in saturation model (default: 1.0)
            random_seed: Random seed for reproducibility
            Pmax: Maximum probability of nucleation at a site (default: 1.0)
            nucleation_sparse_sampling: Fraction of potential nucleation sites to sample
            sort_by_concentration: Whether to sort oligomers by concentration in print_system_state
            min_overlap: Minimum overlap required for annealing
            genome_viz_verbose: Visualization verbosity: 'off' (default), 'every_cycle', or 'every_step'
            max_oligos: Maximum number of oligomers to display in visualizations
            state_save_path: Optional path to save system state to a pickle file after all cycles
            visualization_threshold: Threshold for displaying oligomers in visualizations
            stability_threshold: Threshold for early termination based on concentration stability
            annealing_verbose: Print detailed annealing information
            use_stalling: Use stalling extension function instead of normal extension
            nonstalling_fraction: Fraction of unstalled strands (used when use_stalling=True)
            output_dir: Directory to save output files (default: 'plots')
                
        Returns:
            List of cycle statistics dictionaries
        """
        import time
        import os

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Store output directory for use in other methods
        self._output_dir = output_dir

        # Initialize cycle history if it doesn't exist
        if not hasattr(self, 'cycle_history'):
            self.cycle_history = {
                'total_oligos': [],
                'single_stranded': [],
                'duplexes': [],
                'unique_products': set(),
                'mutant_oligos': [],
                'duplexes_by_mutation': []
            }
        
        # Initialize cycle_states list to store exact state after each cycle
        self.cycle_states = []
        
        # Initialize timing stats dictionary
        timing_stats = {
            'build_reaction_network': [],
            'annealing': [],
            'extension': [],
            'nucleation': [],
            'melting': [],
            'cleanup': [],
            'state_capture': [],
            'total_cycle': []
        }
        
        # Capture initial state (Cycle 0)
        start_capture = time.time()
        self._capture_current_state("initial")
        timing_stats['state_capture'].append(time.time() - start_capture)
            
        cycle_stats = []
            
        # define text file name to save the genome state to after each cycle
        text_genome_filename = os.path.join(output_dir, f'text_genomes_{datetime.now().strftime("%H%M_%m%d")}.txt')
        # Write genome visualization to text file
        original_stdout = sys.stdout  # Save original stdout
        with open(text_genome_filename, 'a') as f:
            sys.stdout = f
            print(f"\n\n{'='*20} PCR CYCLE 0 {'='*20}\n")
            self.print_system_state(visualization_threshold=visualization_threshold, max_oligos=max_oligos, visualize_genome_only=True,  sort_by_concentration=sort_by_concentration)
            sys.stdout = original_stdout  # Restore original stdout


        for cycle in range(1, cycles+1):
            cycle_start_time = time.time()
            
            if verbose:
                print(f"\n\n{'='*20} PCR CYCLE {cycle} {'='*20}\n")
            else:
                print(f"\n--- PCR CYCLE {cycle} ---")
            
            # Run a single PCR cycle
            
            # 1. Annealing phase
            if verbose:
                print("\n========== PCR CYCLE: ANNEALING PHASE ==========")
            else:
                print("\n ==== Annealing phase =====")
            
            # Update the reaction network with any new oligomers
            start_build = time.time()
            self.build_reaction_network(rebuild=True, verbose=verbose, min_overlap=min_overlap)
            timing_stats['build_reaction_network'].append(time.time() - start_build)

            # Store concentrations before annealing
            concentrations_before = np.copy(self.concentrations)
            oligo_indices_before = np.array(range(len(self.concentrations)))
            
            # Simulate the annealing process
            start_annealing = time.time()
            annealing_result = self.simulate_RK45(t_end=annealing_time, verbose=verbose, cycle_num=cycle, 
                                           cleanup_threshold=cleanup_threshold, stability_threshold=stability_threshold)
            timing_stats['annealing'].append(time.time() - start_annealing)
            
            # Plot trajectories if verbose
            if annealing_verbose:
                self.plot_annealing_trajectories(annealing_result.t, annealing_result.y, cycle_num=cycle)


            # Identify oligomers with minimal concentration changes during annealing
            # Calculate percent change for each oligo with concentration > 0.1

            if verbose:
                print("\n----- Oligomers with Minimal Concentration Change During Annealing -----")
                significant_oligos = np.where(concentrations_before > 0.1)[0]
                
                if len(significant_oligos) > 0:
                    percent_changes = np.abs((self.concentrations[significant_oligos] - concentrations_before[significant_oligos]) / 
                                            concentrations_before[significant_oligos] * 100)
                    
                    # Find oligos with change less than 0.01%
                    stable_indices = significant_oligos[percent_changes < 0.01]
                    
                    if len(stable_indices) > 0:
                        print(f"Found {len(stable_indices)} oligomers with concentration >0.1 that changed <0.01% during annealing:")
                        
                        [print(f"  {'Duplex' if self.current_oligomers[i][3] else 'SS'} {i}: {o[0]}-{o[1]} ({'CW' if o[2] else 'CCW'}), Conc: {cb:.6f} → {ca:.6f} ({abs((ca-cb)/cb*100):.6f}%)") for i,cb,ca in zip(stable_indices, concentrations_before[stable_indices], self.concentrations[stable_indices]) if (o:=self.current_oligomers[i])]
                        
                        # Check if stable oligomers are reactants in reaction_pairs
                        print("\nStable oligomers participating in reactions:")
                        reactant_count = 0
                        for stable_idx in stable_indices:
                            # Find reaction pairs where this stable oligomer is a reactant
                            for pair_idx, (r1, r2, _) in enumerate(self.reaction_pairs):
                                if stable_idx == r1 or stable_idx == r2:
                                    # Get the other reactant
                                    other_idx = r2 if stable_idx == r1 else r1
                                    if other_idx < len(self.current_oligomers):
                                        o_other = self.current_oligomers[other_idx]
                                        reactant_count += 1
                                        print(f"  Stable oligo {stable_idx} reacts with oligo {other_idx}: "
                                              f"{o_other[0]}-{o_other[1]} ({'CW' if o_other[2] else 'CCW'}), "
                                              f"Conc: {self.concentrations[other_idx]:.6f}")
                        
                        if reactant_count == 0:
                            print("  None of the stable oligomers are participating in reactions.")
                        
                        print(f"\n\n")
                    else:
                        print("No oligomers with concentration >0.1 had <0.01% change during annealing.")
                else:
                    print("No oligomers with concentration >0.1 found before annealing.")
            
            if verbose:
                self.print_system_state(visualization_threshold=visualization_threshold,  max_oligos=max_oligos, multi_track=False,justtable=False, sort_by_concentration=sort_by_concentration)
            elif genome_viz_verbose == 'every_step':
                self.print_system_state(visualization_threshold=visualization_threshold, max_oligos=max_oligos, visualize_genome_only=True, sort_by_concentration=sort_by_concentration)

            # 2. Extension phase
            if verbose:
                print("\n========== PCR CYCLE: EXTENSION PHASE ==========")
            else:
                print("\n ==== Extension phase =====")
            
            start_extension = time.time()
            if use_stalling:
                extended_oligos = self.extension_oligos_with_stalling(verbose=verbose, min_overlap=min_overlap, nonstalling_fraction=nonstalling_fraction)
            else:
                extended_oligos = self.extension_oligos(verbose=verbose, min_overlap=min_overlap)
            timing_stats['extension'].append(time.time() - start_extension)
            
            if verbose:
                self.print_system_state(visualization_threshold=visualization_threshold,  max_oligos=max_oligos, multi_track=False,justtable=False, sort_by_concentration=sort_by_concentration)
            elif genome_viz_verbose == 'every_step':
                self.print_system_state(visualization_threshold=visualization_threshold,  max_oligos=max_oligos, visualize_genome_only=True, sort_by_concentration=sort_by_concentration)

            # 2.5 De novo nucleation phase (new)
            if enable_nucleation:
                if verbose:
                    print("\n========== PCR CYCLE: DE NOVO NUCLEATION PHASE ==========")
                else:
                    print("\n ==== De novo nucleation phase =====")
                
                start_nucleation = time.time()
                nucleated_oligos = self.nucleate_de_novo_oligos(
                    oligo_length=nucleation_oligo_length,
                    verbose=verbose,
                    seed=random_seed,
                    K=K,
                    Pmax=Pmax,
                    sparse_sampling=nucleation_sparse_sampling,
                    cleanup_threshold=cleanup_threshold
                )
                timing_stats['nucleation'].append(time.time() - start_nucleation)
                
                if verbose:
                    self.print_system_state(visualization_threshold=visualization_threshold, max_oligos=max_oligos, multi_track=False, justtable=False, sort_by_concentration=sort_by_concentration)
                elif genome_viz_verbose == 'every_step':
                    self.print_system_state(visualization_threshold=visualization_threshold, max_oligos=max_oligos, visualize_genome_only=True, sort_by_concentration=sort_by_concentration)
            else:
                timing_stats['nucleation'].append(0)  # Record 0 if nucleation is disabled

            # 3. Melting phase
            if verbose:
                print("\n========== PCR CYCLE: MELTING PHASE ==========")
            else:
                print("\n ==== Melting phase =====")
            
            start_melting = time.time()
            melted_products = self.simulate_melt_cycle(min_stable_length=min_stable_length, verbose=verbose)
            timing_stats['melting'].append(time.time() - start_melting)

            if verbose:
                self.print_system_state(visualization_threshold=visualization_threshold, max_oligos=max_oligos, multi_track=False, justtable=False, sort_by_concentration=sort_by_concentration)
            elif genome_viz_verbose == 'every_step':
                self.print_system_state(visualization_threshold=visualization_threshold, max_oligos=max_oligos, visualize_genome_only=True, sort_by_concentration=sort_by_concentration)

            # 4. Clean up the system
            if verbose:
                print("\n========== PCR CYCLE: CLEANUP ==========")
            else:
                print("\n ==== Cleanup phase =====")
            
            start_cleanup = time.time()
            removed_count = self.cleanup_oligomers(cleanup_threshold)
            #removed_count = 0
            timing_stats['cleanup'].append(time.time() - start_cleanup)

            # Print entire state
            if verbose:
                self.print_system_state(visualization_threshold=visualization_threshold, max_oligos=max_oligos, multi_track=False, justtable=False, sort_by_concentration=sort_by_concentration)  # Customized
            elif genome_viz_verbose == 'every_cycle' or genome_viz_verbose == 'every_step':
                self.print_system_state(visualization_threshold=visualization_threshold, max_oligos=max_oligos, visualize_genome_only=True, sort_by_concentration=sort_by_concentration)

            # Compute oligo length histograms and allele concentrations with integrated printing
            if verbose:
                self.compute_oligo_length_histograms(visualization_threshold=visualization_threshold, print_results=True)
            self.compute_allele_concentrations(print_results=True)

            # Capture the state after this cycle
            start_capture = time.time()
            self._capture_current_state(f"cycle_{cycle}")
            timing_stats['state_capture'].append(time.time() - start_capture)
            
            
            # Write genome visualization to text file
            original_stdout = sys.stdout  # Save original stdout
            with open(text_genome_filename, 'a') as f:
                sys.stdout = f
                print(f"\n\n{'='*20} PCR CYCLE {cycle} {'='*20}\n")
                self.print_system_state(visualization_threshold=visualization_threshold, max_oligos=max_oligos, visualize_genome_only=True, sort_by_concentration=sort_by_concentration)
                sys.stdout = original_stdout  # Restore original stdout
            
            # Record total cycle time
            cycle_time = time.time() - cycle_start_time
            timing_stats['total_cycle'].append(cycle_time)
            
            # Collect cycle statistics
            mutant_oligos = [oligo for oligo, _ in self.get_single_stranded_oligos(0) if oligo[4] == 1]
            mutant_count = len(mutant_oligos)
            
            # Count duplexes by mutation status
            duplexes_by_mutation = {0: 0, 1: 0, 2: 0, 3: 0}  # Initialize counters for each mutation type
            for _, duplex_tuple, _ in self.get_duplexes(visualization_threshold):
                mutation_status = duplex_tuple[4]
                duplexes_by_mutation[mutation_status] = duplexes_by_mutation.get(mutation_status, 0) + 1
            
            stats = {
                'initial_oligo_count': len(self.current_oligomers),
                'reaction_count': len(self.reaction_pairs),
                'duplexes_formed': len(self.get_duplexes(visualization_threshold)),
                'single_stranded_count': len(self.get_single_stranded_oligos(0)),
                'mutant_oligos_count': mutant_count,
                'duplexes_by_mutation': duplexes_by_mutation,
                'removed_oligomers': removed_count,
                'unique_products': len(self.cycle_history['unique_products']),
                'total_concentration': np.sum(self.concentrations),
                'timing': {
                    'build_reaction_network': timing_stats['build_reaction_network'][-1],
                    'annealing': timing_stats['annealing'][-1],
                    'extension': timing_stats['extension'][-1],
                    'nucleation': timing_stats['nucleation'][-1],
                    'melting': timing_stats['melting'][-1],
                    'cleanup': timing_stats['cleanup'][-1],
                    'state_capture': timing_stats['state_capture'][-1],
                    'total': cycle_time
                }
            }
            cycle_stats.append(stats)
            
            # Update cycle history
            self.cycle_history['total_oligos'].append(len(self.current_oligomers))
            self.cycle_history['single_stranded'].append(len(self.get_single_stranded_oligos(0)))
            self.cycle_history['duplexes'].append(len(self.get_duplexes()))
            self.cycle_history['mutant_oligos'].append(mutant_count)
            self.cycle_history['duplexes_by_mutation'].append(duplexes_by_mutation)
            
            # Print timing information for this cycle
            print("\n--- CYCLE TIMING INFORMATION ---")
            print(f"Build reaction network: {timing_stats['build_reaction_network'][-1]:.2f}s")
            print(f"Annealing simulation:   {timing_stats['annealing'][-1]:.2f}s")
            print(f"Extension:              {timing_stats['extension'][-1]:.2f}s")
            print(f"Nucleation:             {timing_stats['nucleation'][-1]:.2f}s")
            print(f"Melting:                {timing_stats['melting'][-1]:.2f}s")
            print(f"Cleanup:                {timing_stats['cleanup'][-1]:.2f}s")
            print(f"State capture:          {timing_stats['state_capture'][-1]:.2f}s")
            print(f"Total cycle time:       {cycle_time:.2f}s")
            
            # Summary for this cycle
            if verbose:
                print("\n========== PCR CYCLE SUMMARY ==========")
                print(f"Initial oligomer types: {stats['initial_oligo_count']}")
                print(f"Reaction pairs: {stats['reaction_count']}")
                print(f"Duplexes formed: {stats['duplexes_formed']}")
                print(f"Single-stranded oligomers: {stats['single_stranded_count']}")
                print(f"Mutant single-stranded oligomers: {stats['mutant_oligos_count']}")
                print(f"Duplexes by mutation status: {stats['duplexes_by_mutation']}")
                print(f"Oligomers removed in cleanup: {stats['removed_oligomers']}")
                print(f"Unique extended products: {stats['unique_products']}")
                print(f"Total concentration: {stats['total_concentration']:.2f}")
            else:
                print(f"Cycle {cycle} complete: {stats['single_stranded_count']} singles, {stats['duplexes_formed']} duplexes, {stats['mutant_oligos_count']} mutants")
        
        # Final summary at the end of all cycles
        if not verbose:
            print("\nPCR protocol completed:")
            print(f"Cycles: {len(cycle_stats)}")
            print(f"Final oligomer types: {cycle_stats[-1]['initial_oligo_count']}")
            print(f"Final single-stranded: {cycle_stats[-1]['single_stranded_count']}")
            print(f"Final duplexes: {cycle_stats[-1]['duplexes_formed']}")
            print(f"Mutant oligos: {cycle_stats[-1]['mutant_oligos_count']}")
        
        # Print overall timing statistics
        print("\n=== OVERALL TIMING STATISTICS ===")
        avg_times = {
            'build_reaction_network': sum(timing_stats['build_reaction_network']) / len(timing_stats['build_reaction_network']),
            'annealing': sum(timing_stats['annealing']) / len(timing_stats['annealing']),
            'extension': sum(timing_stats['extension']) / len(timing_stats['extension']),
            'nucleation': sum(timing_stats['nucleation']) / len(timing_stats['nucleation']),
            'melting': sum(timing_stats['melting']) / len(timing_stats['melting']),
            'cleanup': sum(timing_stats['cleanup']) / len(timing_stats['cleanup']),
            'state_capture': sum(timing_stats['state_capture'][1:]) / (len(timing_stats['state_capture']) - 1),  # Skip initial state
            'total': sum(timing_stats['total_cycle']) / len(timing_stats['total_cycle'])
        }
        
        print(f"Average build reaction network time: {avg_times['build_reaction_network']:.2f}s ({(avg_times['build_reaction_network']/avg_times['total']*100):.1f}%)")
        print(f"Average annealing simulation time:   {avg_times['annealing']:.2f}s ({(avg_times['annealing']/avg_times['total']*100):.1f}%)")
        print(f"Average extension time:              {avg_times['extension']:.2f}s ({(avg_times['extension']/avg_times['total']*100):.1f}%)")
        print(f"Average nucleation time:             {avg_times['nucleation']:.2f}s ({(avg_times['nucleation']/avg_times['total']*100):.1f}%)")
        print(f"Average melting time:                {avg_times['melting']:.2f}s ({(avg_times['melting']/avg_times['total']*100):.1f}%)")
        print(f"Average cleanup time:                {avg_times['cleanup']:.2f}s ({(avg_times['cleanup']/avg_times['total']*100):.1f}%)")
        print(f"Average state capture time:          {avg_times['state_capture']:.2f}s ({(avg_times['state_capture']/avg_times['total']*100):.1f}%)")
        print(f"Average total cycle time:            {avg_times['total']:.2f}s")
        
        # Create visualizations if requested
        if save_plots:
            self.visualize_pcr_results(save_plots=True, show_plots=False, output_dir=output_dir)

        # Save system state if requested
        if state_save_path:
            try:
                # Try to use the save_system_state method if available (in ExtendedOligomerSystem)
                if hasattr(self, 'save_system_state'):
                    print(f"Saving system state to {state_save_path}")
                    self.save_system_state(state_save_path)
            except Exception as e:
                print(f"Error saving system state: {e}")

        return cycle_stats
   
    def _find_contiguous_regions(self, positions):
        """
        Find contiguous regions in a set of positions.
        
        Args:
            positions: Set of positions
            
        Returns:
            List of (start, end) tuples for contiguous regions
        """
        if not positions:
            return []
        
        # Convert to sorted list
        pos_list = sorted(positions)
        
        # Handle circular genome by checking if the first and last positions are adjacent
        if (pos_list[-1] + 1) % self.genome_length == pos_list[0]:
            # Rotate the list to start with the smallest gap
            min_gap = self.genome_length
            min_gap_idx = 0
            
            for i in range(len(pos_list)):
                next_idx = (i + 1) % len(pos_list)
                gap = (pos_list[next_idx] - pos_list[i]) % self.genome_length
                if gap > 1 and gap < min_gap:
                    min_gap = gap
                    min_gap_idx = next_idx
            
            # Rotate the list
            pos_list = pos_list[min_gap_idx:] + pos_list[:min_gap_idx]
        
        # Find contiguous regions
        regions = []
        region_start = pos_list[0]
        prev_pos = pos_list[0]
        
        for pos in pos_list[1:]:
            if pos != (prev_pos + 1) % self.genome_length:
                # End of a contiguous region
                regions.append((region_start, prev_pos))
                region_start = pos
            prev_pos = pos
        
        # Add the last region
        regions.append((region_start, prev_pos))
        
        return regions
    
    def _region_length(self, start, end):
        """
        Calculate the length of a region, handling circular genome.
        
        Args:
            start: Start position
            end: End position
            
        Returns:
            Length of the region
        """
        if end >= start:
            return end - start + 1
        return (end + self.genome_length) - start + 1
    
    def calculate_annealing_length(self, oligo1, oligo2):
        """
        Calculate the annealing length between two oligomers using the efficient BitArray method.
        
        Args:
            oligo1: Tuple (start, end, is_clockwise, is_duplex, is_mutant, [reactant1_idx, reactant2_idx]) for first oligomer
            oligo2: Tuple (start, end, is_clockwise, is_duplex, is_mutant, [reactant1_idx, reactant2_idx]) for second oligomer
                
        Returns:
            Integer representing the number of overlapping positions between the oligomers
        """
        _, overlap_size = self.calculate_overlap(oligo1, oligo2)
        return overlap_size

# Add JIT-compiled functions at module level
if NUMBA_AVAILABLE:
    @njit(fastmath=True)
    def _numba_reaction_rates(concentrations, reaction_pairs_array):
        """
        JIT-compiled version of reaction_rates for better performance.
        
        Args:
            concentrations: Array of current concentrations
            reaction_pairs_array: NumPy array of reaction pair indices
            
        Returns:
            Array of reaction rates
        """
        # Allocate output array
        n_reactions = len(reaction_pairs_array)
        rates = np.empty(n_reactions, dtype=np.float64)
        
        # Fixed rate constant
        k = 1.0
        
        # Compute rates for all reactions
        for i in range(n_reactions):
            reactant1_idx = reaction_pairs_array[i, 0]
            reactant2_idx = reaction_pairs_array[i, 1]
            rates[i] = k * concentrations[reactant1_idx] * concentrations[reactant2_idx]
            
        return rates

    @njit(fastmath=True)
    def _numba_sparse_matmul(rates, data, indices, indptr, shape):
        """
        JIT-compiled sparse matrix multiplication for CSR format.
        Computes the equivalent of sparse_matrix @ rates.
        
        Args:
            rates: Rate vector
            data, indices, indptr: CSR matrix components
            shape: Matrix shape as tuple
            
        Returns:
            Result vector
        """
        m = shape[0]  # Number of rows
        result = np.zeros(m, dtype=np.float64)
        
        # Perform CSR matrix multiplication
        for i in range(m):
            row_start = indptr[i]
            row_end = indptr[i + 1]
            
            for j in range(row_start, row_end):
                col = indices[j]
                if col < len(rates):  # Bounds check
                    result[i] += data[j] * rates[col]
                    
        return result
