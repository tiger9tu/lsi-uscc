#!/usr/bin/env python3
"""
C8 Molecule Energy Comparison Study

Compares the following methods:
1. LASVQENOSI Protocol 1: Fragment-pair specific excitation sets
2. LASVQENOSI Protocol 2: Random division of excitation sets
3. LAS-VQE: VQE over all excitations (single state)
4. CASCI: Complete active space CI reference
5. LASSCF: From LASVQENOSI calculations (no additional computation)

For C8 molecule with fragment spin orbital indices:
- Fragment 0: (0,1,8,9)
- Fragment 1: (2,3,10,11)
- Fragment 2: (4,5,12,13)
- Fragment 3: (6,7,14,15)
"""

import os
import sys
import numpy as np
import random
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from pyscf import gto, scf, mcscf
from scipy.linalg import eigh

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import template and our LAS-VQE-NOSI implementation
from energy_comparison_template import MolecularEnergyComparison, ComparisonResult
from las_vqe_nosi import LASVQENOSI


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


class C8EnergyComparison(MolecularEnergyComparison):
    """C8 molecule energy comparison framework"""
    
    def __init__(self, basis: str = '6-31g', verbose: int = 1):
        """
        Initialize C8 comparison study
        
        Parameters:
        -----------
        basis : str
            Basis set for calculations
        verbose : int
            Verbosity level
        """
        # Initialize parent class
        super().__init__(molecule_name="C8", geom_filename="c8.xyz", basis=basis, verbose=verbose)
        
        # Fragment definitions for C8 from mole_lasinfo.txt
        self.ncas_sub = (2, 2, 2, 2)
        self.nelec_sub = ((1, 1), (1, 1), (1, 1), (1, 1))
        self.spin_sub = (1, 1, 1, 1)
        self.frag_atom_list = [[0, 2], [10, 12], [13, 11], [3, 1]]  # C8 fragment atoms
        
        # Fragment spin orbital indices (after LASSCF orbital ordering)
        self.frag_spin_orbs = {
            0: (0, 1, 8, 9),     # Fragment 0 spin orbitals
            1: (2, 3, 10, 11),   # Fragment 1 spin orbitals  
            2: (4, 5, 12, 13),   # Fragment 2 spin orbitals
            3: (6, 7, 14, 15)    # Fragment 3 spin orbitals
        }
        
        # Setup molecule after configuration
        self.setup_molecule()
        
        # Results storage
        self.results: List[ComparisonResult] = []
        
    
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
    
    def run_las_vqe_protocol1(self, gradient_threshold: float = 0.001, 
                             vqe_max_cycles: int = 100) -> ComparisonResult:
        """
        Protocol 1: Fragment-pair specific excitation sets
        
        Divides excitations into sets based on fragment pairs:
        - Set 1: Excitations within fragment pair (0,1)
        - Set 2: Excitations within fragment pair (1,2)  
        - Set 3: Excitations within fragment pair (2,3)
        """
        if self.verbose >= 1:
            print("\n" + "="*60)
            print("PROTOCOL 1: Fragment-Pair Specific Excitation Sets")
            print("="*60)
        
        start_time = time.time()
        
        try:
            # Setup base calculation
            calc = self._setup_base_calculation(gradient_threshold, vqe_max_cycles)
        
        # Get all excitations using gradient selection
        default_sets = calc.get_default_excitation_sets()
        all_a_idxs = []
        all_i_idxs = []
        for a_set, i_set in default_sets:
            all_a_idxs.extend(a_set)
            all_i_idxs.extend(i_set)
        
        if self.verbose >= 1:
            print(f"Total excitations from gradient selection: {len(all_a_idxs)}")
        
        # Fragment pairs for C8 (linear chain)
        fragment_pairs = [(0, 1), (1, 2), (2, 3)]
        excitation_sets = []
        
        # Get excitations for each fragment pair
        for frag_pair in fragment_pairs:
            pair_a_idxs, pair_i_idxs = self._get_fragment_pair_excitations(
                all_a_idxs, all_i_idxs, frag_pair
            )
            
            if pair_a_idxs:
                excitation_sets.append((pair_a_idxs, pair_i_idxs))
                if self.verbose >= 1:
                    print(f"Fragment pair {frag_pair}: {len(pair_a_idxs)} excitations")
        
        # Fallback if fragment filtering is too restrictive
        if not excitation_sets:
            if self.verbose >= 1:
                print("Fragment filtering too restrictive, using single excitation set")
            # Take top excitations by gradient magnitude
            top_excitations = min(5, len(all_a_idxs))
            excitation_sets = [(all_a_idxs[:top_excitations], all_i_idxs[:top_excitations])]
        
        # Set excitation parameter sets and run VQE
        calc.excitation_parameter_sets = excitation_sets
        vqe_states = calc.perform_vqe_on_sets()
        
        # Perform state interaction if multiple states
        if len(vqe_states) > 1:
            calc.perform_state_interaction()
            final_energy = calc.result.ground_state_energy
            h_matrix = calc.result.hamiltonian_matrix
            s_matrix = calc.result.overlap_matrix
        else:
            final_energy = vqe_states[0].energy if vqe_states else None
            h_matrix = None
            s_matrix = None
        
        calc_time = time.time() - start_time
        
            result = ComparisonResult(
                method="LAS-VQE-NOSI-P1",
                energy=final_energy,
                time=calc_time,
                converged=any([state.converged for state in vqe_states]) if vqe_states else False,
                additional_info={
                    'individual_energies': [state.energy for state in vqe_states] if vqe_states else [],
                    'excitation_sets': len(excitation_sets),
                    'fragment_pairs': fragment_pairs,
                    'n_excitations': sum(len(exc_set[0]) + len(exc_set[1]) for exc_set in excitation_sets),
                    'n_states': len(excitation_sets),
                    'H_matrix': h_matrix,
                    'S_matrix': s_matrix
                }
            )
            
            self.results.append(result)
            
            if self.verbose >= 1:
                print(f"Protocol 1 final energy: {final_energy:.10f} hartree")
                print(f"Calculation time: {calc_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            print(f"LAS-VQE-NOSI Protocol 1 failed: {e}")
            return ComparisonResult(
                method="LAS-VQE-NOSI-P1",
                energy=float('inf'),
                time=time.time() - start_time,
                converged=False,
                additional_info={"error": str(e)}
            )
    
    def run_las_vqe_protocol2(self, gradient_threshold: float = 0.001, 
                             vqe_max_cycles: int = 100) -> ComparisonResult:
        """
        Protocol 2: Random division of excitation sets
        
        Randomly divides all excitations into 3 equal sets for state interaction
        """
        if self.verbose >= 1:
            print("\n" + "="*60)
            print("PROTOCOL 2: Random Division of Excitation Sets")
            print("="*60)
        
        start_time = time.time()
        
        try:
            # Setup base calculation
            calc = self._setup_base_calculation(gradient_threshold, vqe_max_cycles)
        
        # Get all excitations using gradient selection
        default_sets = calc.get_default_excitation_sets()
        all_a_idxs = []
        all_i_idxs = []
        for a_set, i_set in default_sets:
            all_a_idxs.extend(a_set)
            all_i_idxs.extend(i_set)
        
        if self.verbose >= 1:
            print(f"Total excitations from gradient selection: {len(all_a_idxs)}")
        
        # Random division into 3 sets
        random.seed(42)  # For reproducibility
        combined_excitations = list(zip(all_a_idxs, all_i_idxs))
        random.shuffle(combined_excitations)
        
        n_sets = 3
        set_size = len(combined_excitations) // n_sets
        excitation_sets = []
        
        for i in range(n_sets):
            start_idx = i * set_size
            if i == n_sets - 1:  # Last set gets remainder
                end_idx = len(combined_excitations)
            else:
                end_idx = (i + 1) * set_size
            
            set_excitations = combined_excitations[start_idx:end_idx]
            if set_excitations:
                set_a_idxs, set_i_idxs = zip(*set_excitations)
                excitation_sets.append((list(set_a_idxs), list(set_i_idxs)))
                if self.verbose >= 1:
                    print(f"Random set {i+1}: {len(set_a_idxs)} excitations")
        
            if not excitation_sets:
                if self.verbose >= 1:
                    print("No excitations available for Protocol 2")
                return ComparisonResult(
                    method="LAS-VQE-NOSI-P2",
                    energy=None,
                    time=time.time() - start_time,
                    converged=False
                )
        
        # Set excitation parameter sets and run VQE
        calc.excitation_parameter_sets = excitation_sets
        vqe_states = calc.perform_vqe_on_sets()
        
        # Perform state interaction
        if len(vqe_states) > 1:
            calc.perform_state_interaction()
            final_energy = calc.result.ground_state_energy
            h_matrix = calc.result.hamiltonian_matrix
            s_matrix = calc.result.overlap_matrix
        else:
            final_energy = vqe_states[0].energy if vqe_states else None
            h_matrix = None
            s_matrix = None
        
        calc_time = time.time() - start_time
        
            result = ComparisonResult(
                method="LAS-VQE-NOSI-P2",
                energy=final_energy,
                time=calc_time,
                converged=any([state.converged for state in vqe_states]) if vqe_states else False,
                additional_info={
                    'individual_energies': [state.energy for state in vqe_states] if vqe_states else [],
                    'excitation_sets': len(excitation_sets),
                    'random_seed': 42,
                    'n_excitations': sum(len(exc_set[0]) + len(exc_set[1]) for exc_set in excitation_sets),
                    'n_states': len(excitation_sets),
                    'H_matrix': h_matrix,
                    'S_matrix': s_matrix
                }
            )
            
            self.results.append(result)
            
            if self.verbose >= 1:
                print(f"Protocol 2 final energy: {final_energy:.10f} hartree")
                print(f"Calculation time: {calc_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            print(f"LAS-VQE-NOSI Protocol 2 failed: {e}")
            return ComparisonResult(
                method="LAS-VQE-NOSI-P2",
                energy=float('inf'),
                time=time.time() - start_time,
                converged=False,
                additional_info={"error": str(e)}
            )
    
    def run_las_vqe(self) -> ComparisonResult:
        """
        LAS-VQE: Single VQE optimization using all excitations
        """
        if self.verbose >= 1:
            print("\n" + "="*60)
            print("LAS-VQE: All Excitations (Single State)")
            print("="*60)
        
        start_time = time.time()
        
        # Setup base calculation
        calc = self._setup_base_calculation()
        
        # Get all excitations and use them in a single VQE
        default_sets = calc.get_default_excitation_sets()
        all_a_idxs = []
        all_i_idxs = []
        for a_set, i_set in default_sets:
            all_a_idxs.extend(a_set)
            all_i_idxs.extend(i_set)
        
        if self.verbose >= 1:
            print(f"Total excitations for single VQE: {len(all_a_idxs)}")
        
        # Single excitation set
        calc.excitation_parameter_sets = [(all_a_idxs, all_i_idxs)]
        vqe_states = calc.perform_vqe_on_sets()
        
        final_energy = vqe_states[0].energy if vqe_states else None
        calc_time = time.time() - start_time
        
        result = ComparisonResult(
            method_name="LAS-VQE (All Excitations)",
            energy=final_energy,
            n_excitations=len(all_a_idxs) + len(all_i_idxs),
            n_states=1,
            calculation_time=calc_time,
            converged=vqe_states[0].converged if vqe_states else False,
            additional_info={
                'total_excitations': len(all_a_idxs) + len(all_i_idxs)
            }
        )
        
        self.results.append(result)
        
        if self.verbose >= 1:
            print(f"LAS-VQE final energy: {final_energy:.10f} hartree")
            print(f"Calculation time: {calc_time:.2f} seconds")
        
        return result
    
    def run_casci_reference(self) -> ComparisonResult:
        """
        CASCI: Complete active space CI reference
        """
        if self.verbose >= 1:
            print("\n" + "="*60)
            print("CASCI: Complete Active Space CI Reference")
            print("="*60)
        
        start_time = time.time()
        
        # Mean-field calculation
        mf = scf.RHF(self.mol).run()
        
        # CASCI calculation
        ncas_total = sum(self.ncas_sub)
        nelec_total = sum(sum(nelec) for nelec in self.nelec_sub)
        mc_casci = mcscf.CASCI(mf, ncas_total, nelec_total)
        mc_casci.kernel()
        
        calc_time = time.time() - start_time
        
        result = ComparisonResult(
            method_name="CASCI Reference",
            energy=mc_casci.e_tot,
            n_excitations=0,  # Full CI within active space
            n_states=1,
            calculation_time=calc_time,
            converged=mc_casci.converged,
            additional_info={
                'active_space': f"({ncas_total},{nelec_total})",
                'ci_vector_size': mc_casci.ci.size if hasattr(mc_casci.ci, 'size') else 'N/A'
            }
        )
        
        self.results.append(result)
        
        if self.verbose >= 1:
            print(f"CASCI energy: {mc_casci.e_tot:.10f} hartree")
            print(f"Calculation time: {calc_time:.2f} seconds")
        
        return result
    
    def run_complete_comparison(self):
        """Run complete comparison of all methods"""
        
        print("\n" + "="*80)
        print("C8 COMPLETE ENERGY COMPARISON STUDY")
        print("="*80)
        print(f"Basis set: {self.basis}")
        print(f"Gradient threshold: {self.gradient_threshold}")
        print(f"VQE max cycles: {self.vqe_max_cycles}")
        print(f"Molecule: C8 polyene")
        print(f"Active space: {self.ncas_sub} orbitals, {self.nelec_sub} electrons")
        print("="*80)
        
        total_start = time.time()
        
        try:
            # 1. CASCI Reference
            casci_result = self.run_casci_reference()
            
            # 2. LAS-VQE (All Excitations)
            las_vqe_result = self.run_las_vqe()
            
            # 3. Protocol 1 (Fragment-based)
            protocol1_result = self.run_protocol1()
            
            # 4. Protocol 2 (Random division)
            protocol2_result = self.run_protocol2()
            
            # 5. LASSCF (from base calculation)
            # Extract LASSCF energy from one of the calculations
            if hasattr(self, '_base_calc_lasscf_energy'):
                lasscf_result = ComparisonResult(
                    method_name="LASSCF",
                    energy=self._base_calc_lasscf_energy,
                    n_excitations=0,
                    n_states=1,
                    calculation_time=0,  # Included in other calculations
                    converged=True
                )
                self.results.append(lasscf_result)
            
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            import traceback
            traceback.print_exc()
            return
        
        total_time = time.time() - total_start
        
        # Generate summary
        self._generate_summary(total_time)
        
        # Write detailed results
        self._write_detailed_results()
    
    def _generate_summary(self, total_time: float):
        """Generate and print summary of results"""
        
        print("\n" + "="*80)
        print("C8 ENERGY COMPARISON RESULTS SUMMARY")
        print("="*80)
        
        if not self.results:
            print("No results to display")
            return
        
        # Sort results by energy (lowest first)
        sorted_results = sorted(self.results, key=lambda x: x.energy if x.energy is not None else float('inf'))
        
        print(f"{'Method':<30} {'Energy (hartree)':<18} {'ΔE (mEh)':<12} {'Time(s)':<10} {'Conv':<6}")
        print("-" * 80)
        
        casci_energy = next((r.energy for r in self.results if 'CASCI' in r.method_name), None)
        
        for result in sorted_results:
            if result.energy is None:
                continue
                
            delta_e = (result.energy - casci_energy) * 1000 if casci_energy else 0.0
            conv_symbol = "✓" if result.converged else "✗"
            
            print(f"{result.method_name:<30} {result.energy:<18.10f} {delta_e:<12.3f} "
                  f"{result.calculation_time:<10.2f} {conv_symbol:<6}")
        
        print("-" * 80)
        print(f"Total calculation time: {total_time:.2f} seconds")
        
        if casci_energy:
            print(f"CASCI reference energy: {casci_energy:.10f} hartree")
            
            # Find best VQE method
            vqe_results = [r for r in sorted_results if 'VQE' in r.method_name and r.energy is not None]
            if vqe_results:
                best_vqe = vqe_results[0]
                improvement = (casci_energy - best_vqe.energy) * 1000
                print(f"Best VQE improvement: {improvement:.3f} mEh ({best_vqe.method_name})")
    
    def _write_detailed_results(self):
        """Write detailed results to markdown file"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"C8_6-31G_COMPARISON_RESULTS_{timestamp}.md"
        
        with open(filename, 'w') as f:
            f.write(f"# C8 Energy Comparison Results - {self.basis.upper()} Basis\n\n")
            f.write(f"## Overview\n\n")
            f.write(f"**Molecule**: C8 polyene chain\n")
            f.write(f"**Basis**: {self.basis.upper()}\n")
            f.write(f"**Gradient Threshold**: {self.gradient_threshold}\n")
            f.write(f"**VQE Max Cycles**: {self.vqe_max_cycles}\n")
            f.write(f"**Active Space**: {self.ncas_sub} orbitals, {self.nelec_sub} electrons\n\n")
            
            f.write(f"**Fragment Definition**:\n")
            for i, atoms in enumerate(self.frag_atom_list):
                spin_orbs = self.frag_spin_orbs[i]
                f.write(f"- Fragment {i}: atoms {atoms}, spin orbs {spin_orbs}\n")
            f.write("\n")
            
            f.write(f"## Energy Results Summary\n\n")
            f.write(f"| Method | Energy (hartree) | ΔE (mEh) | N_excitations | N_states | Time(s) | Converged |\n")
            f.write(f"|--------|------------------|----------|---------------|----------|---------|-----------|\n")
            
            # Sort results by energy
            sorted_results = sorted(self.results, key=lambda x: x.energy if x.energy is not None else float('inf'))
            casci_energy = next((r.energy for r in self.results if 'CASCI' in r.method_name), None)
            
            for result in sorted_results:
                if result.energy is None:
                    continue
                    
                delta_e = (result.energy - casci_energy) * 1000 if casci_energy else 0.0
                conv_symbol = "✓" if result.converged else "✗"
                
                f.write(f"| **{result.method_name}** | {result.energy:.10f} | {delta_e:.3f} | "
                       f"{result.n_excitations} | {result.n_states} | {result.calculation_time:.2f} | {conv_symbol} |\n")
            
            # Add detailed analysis sections
            f.write(f"\n## Detailed Analysis\n\n")
            
            for result in self.results:
                if 'Protocol' in result.method_name and result.additional_info:
                    f.write(f"### {result.method_name}\n\n")
                    
                    if 'individual_energies' in result.additional_info:
                        f.write(f"**Individual VQE Energies**:\n")
                        for i, energy in enumerate(result.additional_info['individual_energies']):
                            f.write(f"- State {i+1}: {energy:.8f} hartree\n")
                        f.write(f"\n**Final Energy After State Interaction**: {result.energy:.10f} hartree\n\n")
                    
                    if result.H_matrix is not None:
                        f.write(f"**Hamiltonian Matrix (hartree)**:\n```\n")
                        for row in result.H_matrix:
                            f.write("[" + "  ".join(f"{val:.8f}" for val in row) + "]\n")
                        f.write("```\n\n")
                    
                    if result.S_matrix is not None:
                        f.write(f"**Overlap Matrix**:\n```\n")
                        for row in result.S_matrix:
                            f.write("[" + "  ".join(f"{val:.8f}" for val in row) + "]\n")
                        f.write("```\n\n")
            
            f.write(f"## Conclusions\n\n")
            f.write(f"This C8 energy comparison demonstrates the performance of LAS-VQE-NOSI methods ")
            f.write(f"on an 8-carbon polyene system with {self.basis.upper()} basis set.\n\n")
            
            if casci_energy:
                best_method = min(self.results, key=lambda x: x.energy if x.energy else float('inf'))
                if best_method.energy:
                    accuracy = (best_method.energy - casci_energy) * 1000
                    f.write(f"**Best accuracy**: {best_method.method_name} achieves {accuracy:.3f} mEh from CASCI reference.\n\n")
        
        print(f"Detailed results written to: {filename}")


def main():
    """Main execution"""
    # Create comparison with 6-31g basis and production parameters
    comparison = C8EnergyComparison(
        basis='6-31g',
        gradient_threshold=0.0001,
        vqe_max_cycles=100,
        verbose=1
    )
    
    # Run complete comparison
    comparison.run_complete_comparison()

if __name__ == "__main__":
    main()