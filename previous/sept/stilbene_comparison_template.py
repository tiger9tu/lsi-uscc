#!/usr/bin/env python3
"""
Stilbene Energy Comparison Template - LAS-VQE-NOSI Methods
Template for stilbene conformer energy comparison studies
"""

import os
import sys
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import template and our LAS-VQE-NOSI implementation
from energy_comparison_template import MolecularEnergyComparison, ComparisonResult, run_standard_test_suite
from las_vqe_nosi import LASVQENOSI

class StilbeneEnergyComparison(MolecularEnergyComparison):
    """Base class for stilbene conformer energy comparison calculations"""
    
    def __init__(self, conformer_name: str, geom_filename: str, basis: str = '6-31g', verbose: int = 1):
        # Initialize parent class
        super().__init__(molecule_name=f"Stilbene-{conformer_name}", geom_filename=geom_filename, basis=basis, verbose=verbose)
        
        # Stilbene molecular configuration (same for all conformers)
        self.ncas_sub = (4, 2, 4)
        self.nelec_sub = ((2, 2), (1, 1), (2, 2))
        self.spin_sub = (1, 1, 1)
        self.frag_atom_list = [[1,2,3,4,5,6,15,16,17,18,19], [0,7,14,20], [8,9,10,11,12,13,21,22,23,24,25]]
        
        # Fragment spin orbital indices for Stilbene (same for all conformers)
        self.frag_spin_orbs = {
            0: (0, 1, 2, 3, 10, 11, 12, 13),  # First phenyl ring
            1: (4, 5, 14, 15),                # Vinyl bridge
            2: (6, 7, 8, 9, 16, 17, 18, 19)   # Second phenyl ring
        }
        
        self.conformer_name = conformer_name
        
        # Setup molecule after configuration
        self.setup_molecule()
    
    def run_las_vqe_protocol1(self, gradient_threshold: float = 0.001, 
                             vqe_max_cycles: int = 100) -> ComparisonResult:
        """Run LAS-VQE-NOSI Protocol 1 (fragment-based excitations)"""
        print(f"\nRunning LAS-VQE-NOSI Protocol 1 (threshold={gradient_threshold}, cycles={vqe_max_cycles})...")
        start_time = time.time()
        
        try:
            # Setup calculation
            las_calc = LASVQENOSI(
                mol=self.mol,
                ncas_sub=self.ncas_sub,
                nelec_sub=self.nelec_sub,
                basis=self.basis,
                frag_atom_list=self.frag_atom_list,
                gradient_threshold=gradient_threshold,
                vqe_max_cycles=vqe_max_cycles,
                verbose=self.verbose
            )
            
            # Setup basic calculations
            las_calc.setup_molecule()
            las_calc.perform_lasscf()
            las_calc.setup_casci_reference()
            
            # Protocol 1: Fragment-based excitation selection
            # Generate excitations based on fragment pairs
            excitations = {}
            for state_idx in range(len(self.frag_spin_orbs)):
                state_excitations = []
                
                # Fragment pairs for stilbene: (0,1), (1,2), (0,2)
                frag_pairs = [(0, 1), (1, 2), (0, 2)]
                
                for frag_i, frag_j in frag_pairs:
                    if frag_i in self.frag_spin_orbs and frag_j in self.frag_spin_orbs:
                        frag_i_orbs = self.frag_spin_orbs[frag_i]
                        frag_j_orbs = self.frag_spin_orbs[frag_j]
                        
                        # Add representative excitations between fragments
                        # For stilbene, focus on π-π* transitions
                        for orb_i in frag_i_orbs[:4]:  # Limit to first 4 orbs per fragment
                            for orb_j in frag_j_orbs[:4]:
                                if abs(orb_i - orb_j) >= 2:  # Avoid same-orbital excitations
                                    state_excitations.append((orb_i, orb_j))
                
                # Add intra-fragment excitations for large fragments (phenyl rings)
                if state_idx in [0, 2]:  # Phenyl fragments
                    frag_orbs = self.frag_spin_orbs[state_idx]
                    if len(frag_orbs) >= 6:
                        state_excitations.append((frag_orbs[0], frag_orbs[4]))
                        state_excitations.append((frag_orbs[1], frag_orbs[5]))
                
                # Limit total excitations per state
                if len(state_excitations) > 12:
                    state_excitations = state_excitations[:12]
                
                excitations[state_idx] = state_excitations
            
            # Assign excitations to calculation object
            las_calc.excitations = excitations
            
            # Run calculation
            las_calc.run()
            
            calculation_time = time.time() - start_time
            
            return ComparisonResult(
                method="LAS-VQE-NOSI-P1",
                energy=las_calc.final_energy,
                time=calculation_time,
                converged=las_calc.converged,
                additional_info={
                    "individual_energies": las_calc.vqe_energies.copy(),
                    "excitations_count": sum(len(exc) for exc in excitations.values()),
                    "threshold": gradient_threshold,
                    "max_cycles": vqe_max_cycles
                }
            )
            
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
        """Run LAS-VQE-NOSI Protocol 2 (broader excitation selection)"""
        print(f"\nRunning LAS-VQE-NOSI Protocol 2 (threshold={gradient_threshold}, cycles={vqe_max_cycles})...")
        start_time = time.time()
        
        try:
            # Setup calculation
            las_calc = LASVQENOSI(
                mol=self.mol,
                ncas_sub=self.ncas_sub,
                nelec_sub=self.nelec_sub,
                basis=self.basis,
                frag_atom_list=self.frag_atom_list,
                gradient_threshold=gradient_threshold,
                vqe_max_cycles=vqe_max_cycles,
                verbose=self.verbose
            )
            
            # Setup basic calculations
            las_calc.setup_molecule()
            las_calc.perform_lasscf()
            las_calc.setup_casci_reference()
            
            # Protocol 2: More systematic excitation selection
            excitations = {}
            all_orb_indices = []
            for frag_orbs in self.frag_spin_orbs.values():
                all_orb_indices.extend(frag_orbs)
            
            for state_idx in range(len(self.frag_spin_orbs)):
                state_excitations = []
                
                # Generate excitations within and between all fragments
                # For stilbene, emphasize π-conjugation pathways
                for i, orb_i in enumerate(all_orb_indices[:12]):  # Limit orbital range
                    for j, orb_j in enumerate(all_orb_indices[i+1:14]):  # Cross-fragment pairs
                        if abs(orb_i - orb_j) >= 2:  # Avoid same-orbital excitations
                            state_excitations.append((orb_i, orb_j))
                
                # Add specific phenyl-vinyl-phenyl conjugation excitations
                phenyl1_orbs = self.frag_spin_orbs[0][:6]  # First phenyl ring
                vinyl_orbs = self.frag_spin_orbs[1]       # Vinyl bridge  
                phenyl2_orbs = self.frag_spin_orbs[2][:6] # Second phenyl ring
                
                # Phenyl-vinyl excitations
                for p_orb in phenyl1_orbs[:3]:
                    for v_orb in vinyl_orbs:
                        state_excitations.append((p_orb, v_orb))
                
                for v_orb in vinyl_orbs:
                    for p_orb in phenyl2_orbs[:3]:
                        state_excitations.append((v_orb, p_orb))
                
                # Limit total excitations per state
                if len(state_excitations) > 18:
                    state_excitations = state_excitations[:18]
                
                excitations[state_idx] = state_excitations
            
            # Assign excitations to calculation object
            las_calc.excitations = excitations
            
            # Run calculation
            las_calc.run()
            
            calculation_time = time.time() - start_time
            
            return ComparisonResult(
                method="LAS-VQE-NOSI-P2",
                energy=las_calc.final_energy,
                time=calculation_time,
                converged=las_calc.converged,
                additional_info={
                    "individual_energies": las_calc.vqe_energies.copy(),
                    "excitations_count": sum(len(exc) for exc in excitations.values()),
                    "threshold": gradient_threshold,
                    "max_cycles": vqe_max_cycles
                }
            )
            
        except Exception as e:
            print(f"LAS-VQE-NOSI Protocol 2 failed: {e}")
            return ComparisonResult(
                method="LAS-VQE-NOSI-P2",
                energy=float('inf'),
                time=time.time() - start_time,
                converged=False,
                additional_info={"error": str(e)}
            )

def create_stilbene_class(conformer_name: str, geom_filename: str):
    """Factory function to create stilbene conformer classes"""
    class SpecificStilbeneComparison(StilbeneEnergyComparison):
        def __init__(self, basis: str = '6-31g', verbose: int = 1):
            super().__init__(conformer_name, geom_filename, basis, verbose)
    
    return SpecificStilbeneComparison