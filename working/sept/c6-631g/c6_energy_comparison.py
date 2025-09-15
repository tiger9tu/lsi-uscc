#!/usr/bin/env python3
"""
C6 Molecule Energy Comparison Study (6-31G basis)

Compares the following methods:
1. LASVQENOSI Protocol 1: Fragment-pair specific excitation sets
2. LASVQENOSI Protocol 2: Random division of excitation sets
3. LAS-VQE: VQE over all excitations (single state)
4. CASCI: Complete active space CI reference
5. LASSCF: From LASVQENOSI calculations (no additional computation)

For C6 molecule with fragment spin orbital indices:
- Fragment 0: (0,1,6,7)
- Fragment 1: (2,3,8,9)
- Fragment 2: (4,5,10,11)
"""

import numpy as np
import random
import time
import os
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from pyscf import gto, scf, mcscf
from scipy.linalg import eigh

# Import our LAS-VQE-NOSI implementation
from las_vqe_nosi import LASVQENOSI
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.fci import csf_solver
from mrh.exploratory.citools import grad
from mrh.exploratory.unitary_cc import lasuccsd


@dataclass
class ComparisonResult:
    """Container for energy comparison results"""
    method_name: str
    energy: float
    n_excitations: int = 0
    n_states: int = 0
    H_matrix: np.ndarray = None
    S_matrix: np.ndarray = None
    min_diag_H: float = None
    calculation_time: float = 0.0
    converged: bool = False
    additional_info: Dict[str, Any] = None


class C6EnergyComparison:
    """C6 molecule energy comparison framework (6-31G basis)"""
    
    def __init__(self, basis: str = '6-31g', gradient_threshold: float = 0.01, 
                 vqe_max_cycles: int = 10, verbose: int = 1):
        """
        Initialize C6 comparison study
        
        Parameters:
        -----------
        basis : str
            Basis set for calculations
        gradient_threshold : float
            Gradient threshold for excitation selection
        vqe_max_cycles : int
            Maximum VQE optimization cycles
        verbose : int
            Verbosity level
        """
        self.basis = basis
        self.gradient_threshold = gradient_threshold
        self.vqe_max_cycles = vqe_max_cycles
        self.verbose = verbose
        
        # C6 molecule setup
        self.c6_geometry = self._load_c6_geometry()
        self.mol = self._setup_molecule()
        
        # Fragment definitions for C6
        self.ncas_sub = (2, 2, 2)
        self.nelec_sub = ((1, 1), (1, 1), (1, 1))
        self.frag_atom_list = ((0, 2), (10, 11), (3, 1))  # C6 fragment atoms
        
        # Fragment spin orbital indices (after LASSCF orbital ordering)
        self.frag_spin_orbs = {
            0: (0, 1, 6, 7),    # Fragment 0 spin orbitals
            1: (2, 3, 8, 9),    # Fragment 1 spin orbitals  
            2: (4, 5, 10, 11)   # Fragment 2 spin orbitals
        }
        
        # Results storage
        self.results: List[ComparisonResult] = []
        
    def _load_c6_geometry(self) -> str:
        """Load C6 geometry from data directory with machine-independent path handling"""
        # Try multiple possible paths
        possible_paths = [
            '../../geom/c6.xyz',
            '../geom/c6.xyz',
            os.path.join(os.path.dirname(__file__), '../../geom/c6.xyz'),
            '/home/jinx/repo/qchem/las_uccsd_data/polyenes/geometries/c6.xyz',
            '/home/jinx/repo/qchem/las-uscc-noci-bot/working/geom/c6.xyz'
        ]
        
        for path in possible_paths:
            try:
                with open(path, 'r') as f:
                    return f.read()
            except FileNotFoundError:
                continue
        
        # Fallback C6 geometry if file not found
        return """C -3.075055 0.167498 0.000000
C 3.075055 -0.167498 0.000000
C -1.867388 -0.418911 0.000000
C 1.867388 0.418911 0.000000
H -3.180907 1.249260 0.000000
H 3.180907 -1.249260 0.000000
H -3.991128 -0.413469 0.000000
H 3.991128 0.413469 0.000000
H -1.805249 -1.507080 0.000000
H 1.805249 1.507080 0.000000
C -0.606997 0.297372 0.000000
C 0.606997 -0.297372 0.000000
H -0.661631 1.386399 0.000000
H 0.661631 -1.386399 0.000000"""
    
    def _setup_molecule(self) -> gto.Mole:
        """Setup C6 molecule"""
        mol = gto.M(atom=self.c6_geometry, basis=self.basis, verbose=0)
        mol.build()
        return mol
    
    def _setup_base_calculation(self) -> LASVQENOSI:
        """Setup base LAS-VQE-NOSI calculation (common for all protocols)"""
        calc = LASVQENOSI(
            mol=self.mol,
            ncas_sub=self.ncas_sub,
            nelec_sub=self.nelec_sub,
            basis=self.basis,
            frag_atom_list=self.frag_atom_list,
            gradient_threshold=self.gradient_threshold,
            vqe_max_cycles=self.vqe_max_cycles,
            verbose=self.verbose
        )
        
        # Setup basic calculations (shared by all methods)
        calc.setup_molecule()
        calc.perform_lasscf()
        calc.setup_casci_reference()
        
        return calc
    
    def _get_fragment_pair_excitations(self, all_a_idxs: List, all_i_idxs: List, 
                                     frag_pair: Tuple[int, int]) -> Tuple[List, List]:
        """
        Extract excitations within a specific fragment pair
        
        Parameters:
        -----------
        all_a_idxs, all_i_idxs : List
            All available excitation indices
        frag_pair : Tuple[int, int]
            Fragment pair indices (e.g., (0,1) for fragments 0 and 1)
            
        Returns:
        --------
        Tuple[List, List]
            Filtered excitation indices for the fragment pair
        """
        # Get combined spin orbitals for the fragment pair
        frag_orbs = set(self.frag_spin_orbs[frag_pair[0]] + self.frag_spin_orbs[frag_pair[1]])

        pair_a_idxs = []
        pair_i_idxs = []
        for idx in range(len(all_a_idxs)):
            if (
                all(elem in frag_orbs for elem in all_a_idxs[idx]) and
                all(elem in frag_orbs for elem in all_i_idxs[idx])
            ):
                pair_a_idxs.append(all_a_idxs[idx])
                pair_i_idxs.append(all_i_idxs[idx])
        print("DEBUG: Fragment pair", frag_pair, "excitations found:", len(pair_a_idxs))
        return pair_a_idxs, pair_i_idxs
    
    def run_protocol1(self) -> ComparisonResult:
        """
        Protocol 1: Fragment-pair specific excitation sets
        
        Divides excitations into 3 sets based on fragment pairs:
        - Set 1: Excitations within fragment pair (0,1)
        - Set 2: Excitations within fragment pair (1,2)  
        - Set 3: Excitations within fragment pair (0,2)
        """
        if self.verbose >= 1:
            print("\n" + "="*60)
            print("PROTOCOL 1: Fragment-Pair Specific Excitation Sets")
            print("="*60)
        
        start_time = time.time()
        
        # Setup base calculation
        calc = self._setup_base_calculation()
        
        # Get all excitations using gradient selection
        default_sets = calc.get_default_excitation_sets()
        all_a_idxs = []
        all_i_idxs = []
        for a_set, i_set in default_sets:
            all_a_idxs.extend(a_set)
            all_i_idxs.extend(i_set)
        
        if self.verbose >= 1:
            print(f"Total excitations from gradient selection: {len(all_a_idxs)}")
        
        # Create fragment-pair specific sets
        # Set 1: Fragment pair (0,1) 
        set1_a, set1_i = self._get_fragment_pair_excitations(all_a_idxs, all_i_idxs, (0, 1))
        
        # Set 2: Fragment pair (1,2)
        set2_a, set2_i = self._get_fragment_pair_excitations(all_a_idxs, all_i_idxs, (1, 2))
        
        # Set 3: Fragment pair (0,2) 
        set3_a, set3_i = self._get_fragment_pair_excitations(all_a_idxs, all_i_idxs, (0, 2))
        
        # Filter out empty sets and ensure minimum size
        protocol1_sets = []
        for i, (a_set, i_set) in enumerate([(set1_a, set1_i), (set2_a, set2_i), (set3_a, set3_i)]):
            if len(a_set) > 0 and len(i_set) > 0:
                protocol1_sets.append((a_set, i_set))
                if self.verbose >= 1:
                    print(f"Fragment pair {[(0,1), (1,2), (0,2)][i]}: {len(a_set)} excitations")
        
        if len(protocol1_sets) == 0:
            # Fallback: use first few excitations
            protocol1_sets = [(all_a_idxs[:5], all_i_idxs[:5])]
            if self.verbose >= 1:
                print("No fragment-specific excitations found, using fallback set")
        
        # Assign excitation sets and run calculation
        calc.excitation_parameter_sets = protocol1_sets
        calc.perform_vqe_on_sets()
        
        # Check if we have valid VQE states for state interaction
        valid_states = [state for state in calc.vqe_states if state.psi is not None and state.energy != float('inf')]
        
        if len(valid_states) >= 2:
            final_energies, eigenvectors = calc.perform_state_interaction()
            ground_state_energy = final_energies[0]
            H_matrix = calc.result.hamiltonian_matrix.copy()
            S_matrix = calc.result.overlap_matrix.copy()
            min_diag_H = np.min(np.diag(calc.result.hamiltonian_matrix.real))
        else:
            # Fallback to best individual VQE energy if state interaction fails
            valid_energies = [state.energy for state in valid_states]
            ground_state_energy = min(valid_energies) if valid_energies else float('inf')
            H_matrix = None
            S_matrix = None
            min_diag_H = ground_state_energy
            if self.verbose >= 1:
                print(f"Warning: Only {len(valid_states)} valid VQE states, using best individual energy")
        
        # Extract results
        total_excitations = sum(len(a_set) for a_set, i_set in protocol1_sets)
        
        result = ComparisonResult(
            method_name="LAS-VQE-NOSI Protocol 1",
            energy=ground_state_energy,
            n_excitations=total_excitations,
            n_states=len(protocol1_sets),
            H_matrix=H_matrix,
            S_matrix=S_matrix,
            min_diag_H=min_diag_H,
            calculation_time=time.time() - start_time,
            converged=any(calc.result.vqe_converged) if calc.result.vqe_converged else False,
            additional_info={
                'fragment_pairs': [(0,1), (1,2), (0,2)],
                'set_sizes': [len(a_set) for a_set, i_set in protocol1_sets],
                'individual_vqe_energies': calc.result.individual_vqe_energies.copy(),
                'valid_states': len(valid_states)
            }
        )
        
        self.results.append(result)
        return result
    
    def run_protocol2(self) -> ComparisonResult:
        """
        Protocol 2: Random division of excitation sets
        
        Randomly divides all selected excitations into 3 sets of equal size
        """
        if self.verbose >= 1:
            print("\n" + "="*60)
            print("PROTOCOL 2: Random Division of Excitation Sets")
            print("="*60)
        
        start_time = time.time()
        
        # Setup base calculation
        calc = self._setup_base_calculation()
        
        # Get all excitations
        default_sets = calc.get_default_excitation_sets()
        all_a_idxs = []
        all_i_idxs = []
        for a_set, i_set in default_sets:
            all_a_idxs.extend(a_set)
            all_i_idxs.extend(i_set)
        
        if self.verbose >= 1:
            print(f"Total excitations: {len(all_a_idxs)}")
        
        # Randomly shuffle excitations (with fixed seed for reproducibility)
        random.seed(42)
        combined_excitations = list(zip(all_a_idxs, all_i_idxs))
        random.shuffle(combined_excitations)
        
        # Divide into 3 sets of equal size
        n_total = len(combined_excitations)
        set_size = n_total // 3
        
        protocol2_sets = []
        for i in range(3):
            start_idx = i * set_size
            end_idx = (i + 1) * set_size if i < 2 else n_total  # Last set gets remainder
            
            set_excitations = combined_excitations[start_idx:end_idx]
            if len(set_excitations) > 0:
                a_set, i_set = zip(*set_excitations)
                protocol2_sets.append((list(a_set), list(i_set)))
                
                if self.verbose >= 1:
                    print(f"Random set {i+1}: {len(a_set)} excitations")
        
        # Assign excitation sets and run calculation
        calc.excitation_parameter_sets = protocol2_sets
        calc.perform_vqe_on_sets()
        
        # Check if we have valid VQE states for state interaction
        valid_states = [state for state in calc.vqe_states if state.psi is not None and state.energy != float('inf')]
        
        if len(valid_states) >= 2:
            final_energies, eigenvectors = calc.perform_state_interaction()
            ground_state_energy = final_energies[0]
            H_matrix = calc.result.hamiltonian_matrix.copy()
            S_matrix = calc.result.overlap_matrix.copy()
            min_diag_H = np.min(np.diag(calc.result.hamiltonian_matrix.real))
        else:
            # Fallback to best individual VQE energy if state interaction fails
            valid_energies = [state.energy for state in valid_states]
            ground_state_energy = min(valid_energies) if valid_energies else float('inf')
            H_matrix = None
            S_matrix = None
            min_diag_H = ground_state_energy
            if self.verbose >= 1:
                print(f"Warning: Only {len(valid_states)} valid VQE states, using best individual energy")
        
        # Extract results
        total_excitations = sum(len(a_set) for a_set, i_set in protocol2_sets)
        
        result = ComparisonResult(
            method_name="LAS-VQE-NOSI Protocol 2",
            energy=ground_state_energy,
            n_excitations=total_excitations,
            n_states=len(protocol2_sets),
            H_matrix=H_matrix,
            S_matrix=S_matrix,
            min_diag_H=min_diag_H,
            calculation_time=time.time() - start_time,
            converged=any(calc.result.vqe_converged) if calc.result.vqe_converged else False,
            additional_info={
                'random_seed': 42,
                'set_sizes': [len(a_set) for a_set, i_set in protocol2_sets],
                'individual_vqe_energies': calc.result.individual_vqe_energies.copy(),
                'valid_states': len(valid_states)
            }
        )
        
        self.results.append(result)
        return result
    
    def run_lasvqe_all_excitations(self) -> ComparisonResult:
        """
        LAS-VQE: Single VQE optimization over all selected excitations
        """
        if self.verbose >= 1:
            print("\n" + "="*60)
            print("LAS-VQE: VQE Over All Excitations")
            print("="*60)
        
        start_time = time.time()
        
        # Setup base calculation
        calc = self._setup_base_calculation()
        
        # Get all excitations as a single set
        default_sets = calc.get_default_excitation_sets()
        all_a_idxs = []
        all_i_idxs = []
        for a_set, i_set in default_sets:
            all_a_idxs.extend(a_set)
            all_i_idxs.extend(i_set)
        
        if self.verbose >= 1:
            print(f"Total excitations for single VQE: {len(all_a_idxs)}")
        
        # Single excitation set with all excitations
        calc.excitation_parameter_sets = [(all_a_idxs, all_i_idxs)]
        calc.perform_vqe_on_sets()
        
        # No state interaction needed for single state - just get VQE energy
        vqe_energy = calc.result.individual_vqe_energies[0]
        
        result = ComparisonResult(
            method_name="LAS-VQE (All Excitations)",
            energy=vqe_energy,
            n_excitations=len(all_a_idxs),
            n_states=1,
            H_matrix=None,  # No state interaction
            S_matrix=None,  # No state interaction
            min_diag_H=vqe_energy,  # Single diagonal element
            calculation_time=time.time() - start_time,
            converged=calc.result.vqe_converged[0],
            additional_info={
                'vqe_energy': vqe_energy,
                'note': 'Single VQE state - no state interaction'
            }
        )
        
        self.results.append(result)
        return result
    
    def run_casci_reference(self) -> ComparisonResult:
        """
        CASCI reference calculation
        """
        if self.verbose >= 1:
            print("\n" + "="*60)
            print("CASCI Reference")
            print("="*60)
        
        start_time = time.time()
        
        # Setup molecule and LASSCF for orbital generation
        calc = self._setup_base_calculation()
        
        # CASCI energy is already computed in setup_casci_reference
        casci_energy = calc.casci.e_tot
        
        if self.verbose >= 1:
            print(f"CASCI energy: {casci_energy:.10f} hartree")
        
        result = ComparisonResult(
            method_name="CASCI",
            energy=casci_energy,
            n_excitations=0,  # Full CI in active space
            n_states=1,
            H_matrix=None,
            S_matrix=None,
            min_diag_H=casci_energy,
            calculation_time=time.time() - start_time,
            converged=True,  # CASCI always converges if completed
            additional_info={
                'ncas_total': sum(self.ncas_sub),
                'nelec_total': sum([sum(ne) for ne in self.nelec_sub])
            }
        )
        
        self.results.append(result)
        return result
    
    def get_lasscf_energy(self) -> ComparisonResult:
        """
        Get LASSCF energy (already computed in base calculation)
        """
        if self.verbose >= 1:
            print("\n" + "="*60)
            print("LASSCF Energy (from base calculation)")
            print("="*60)
        
        # Setup just to get LASSCF energy
        calc = self._setup_base_calculation()
        lasscf_energy = calc.las.e_tot
        
        if self.verbose >= 1:
            print(f"LASSCF energy: {lasscf_energy:.10f} hartree")
        
        result = ComparisonResult(
            method_name="LASSCF",
            energy=lasscf_energy,
            n_excitations=0,  # Mean-field reference
            n_states=1,
            H_matrix=None,
            S_matrix=None,
            min_diag_H=lasscf_energy,
            calculation_time=0.0,  # Already computed
            converged=calc.las.converged,
            additional_info={
                'fragments': self.ncas_sub,
                'nelec_fragments': self.nelec_sub
            }
        )
        
        self.results.append(result)
        return result
    
    def run_all_methods(self) -> List[ComparisonResult]:
        """Run all comparison methods"""
        if self.verbose >= 1:
            print(f"\n{'='*80}")
            print("C6 MOLECULE ENERGY COMPARISON STUDY (6-31G BASIS)")
            print(f"Basis: {self.basis}, Gradient threshold: {self.gradient_threshold}")
            print(f"{'='*80}")
        
        # Run all methods
        self.run_protocol1()
        self.run_protocol2()
        self.run_lasvqe_all_excitations()
        self.run_casci_reference()
        self.get_lasscf_energy()
        
        return self.results
    
    def print_comparison_summary(self):
        """Print comprehensive comparison summary"""
        if not self.results:
            print("No results to display")
            return
            
        print(f"\n{'='*100}")
        print("C6 ENERGY COMPARISON SUMMARY (6-31G BASIS)")
        print(f"{'='*100}")
        
        # Main results table
        print(f"{'Method':<25} {'Energy (hartree)':<18} {'ΔE (mEh)':<12} {'N_exc':<8} {'N_states':<10} {'Time (s)':<10} {'Conv':<6}")
        print("-" * 100)
        
        # Find lowest energy for relative comparison
        min_energy = min(r.energy for r in self.results)
        
        for result in self.results:
            delta_e = (result.energy - min_energy) * 1000  # Convert to milliEh
            conv_status = "✓" if result.converged else "✗"
            
            print(f"{result.method_name:<25} {result.energy:<18.10f} {delta_e:<12.3f} "
                  f"{result.n_excitations:<8} {result.n_states:<10} "
                  f"{result.calculation_time:<10.2f} {conv_status:<6}")
        
        print("-" * 100)
        
        # Detailed diagonalization information
        print(f"\nDETAILED DIAGONALIZATION INFORMATION")
        print(f"{'='*60}")
        
        for result in self.results:
            if result.method_name.startswith("LAS-VQE-NOSI"):
                print(f"\n{result.method_name}:")
                print(f"  Total excitations: {result.n_excitations}")
                print(f"  Number of states: {result.n_states}")
                print(f"  Minimum diagonal H: {result.min_diag_H:.10f} hartree")
                
                if result.additional_info:
                    if 'set_sizes' in result.additional_info:
                        print(f"  Set sizes: {result.additional_info['set_sizes']}")
                    if 'individual_vqe_energies' in result.additional_info:
                        individual = result.additional_info['individual_vqe_energies']
                        print(f"  Individual VQE energies: {[f'{e:.8f}' for e in individual]}")
                
                # Print H and S matrices
                if result.H_matrix is not None:
                    print(f"  H matrix:")
                    self._print_matrix(result.H_matrix, "    ")
                    
                if result.S_matrix is not None:
                    print(f"  S matrix:")
                    self._print_matrix(result.S_matrix, "    ")
        
        # Analysis
        print(f"\nENERGY ORDERING ANALYSIS")
        print(f"{'='*40}")
        sorted_results = sorted(self.results, key=lambda x: x.energy)
        for i, result in enumerate(sorted_results):
            print(f"{i+1}. {result.method_name}: {result.energy:.10f} hartree")
        
        # Expected hierarchy check
        print(f"\nMETHOD EXPECTATIONS:")
        print("• CASCI should provide the lowest energy (exact within active space)")
        print("• LAS-VQE-NOSI methods should improve upon LASSCF")
        print("• Protocol 1 (fragment-based) may capture different physics than Protocol 2 (random)")
        print("• Single LAS-VQE should be between LASSCF and CASCI")
    
    def _print_matrix(self, matrix: np.ndarray, indent: str = ""):
        """Print a matrix in readable format"""
        for row in matrix:
            row_str = "  ".join(f"{x.real:+.8f}" if abs(x.imag) < 1e-10 
                              else f"{x.real:+.6f}{x.imag:+.6f}j" for x in row)
            print(f"{indent}[{row_str}]")


def main():
    """Main execution function"""
    
    # Setup comparison study
    comparison = C6EnergyComparison(
        basis='6-31g',
        gradient_threshold=0.0001,
        vqe_max_cycles=30,
        verbose=1
    )
    
    # Run all methods
    results = comparison.run_all_methods()
    
    # Print comprehensive summary
    comparison.print_comparison_summary()
    
    return results


if __name__ == "__main__":
    print("Starting C6 molecule energy comparison study (6-31G basis)...")
    results = main()
    print(f"\nComparison study completed with {len(results)} methods!")