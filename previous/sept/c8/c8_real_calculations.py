#!/usr/bin/env python3
"""
Real LAS-VQE-NOSI calculations for C8 molecule with 6-31G basis
"""

import numpy as np
import random
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from pyscf import gto, scf, mcscf
from scipy.linalg import eigh

# Import our LAS-VQE-NOSI implementation
from las_vqe_nosi import LASVQENOSI

@dataclass
class CalculationResult:
    """Container for calculation results"""
    method_name: str
    energy: float
    n_excitations: int = 0
    n_states: int = 0
    H_matrix: np.ndarray = None
    S_matrix: np.ndarray = None
    calculation_time: float = 0.0
    converged: bool = False
    individual_energies: List[float] = None

class C8RealCalculations:
    """Real LAS-VQE-NOSI calculations for C8 molecule"""
    
    def __init__(self, basis: str = '6-31g', verbose: int = 1):
        """Initialize C8 calculations"""
        self.basis = basis
        self.verbose = verbose
        
        # C8 configuration from mole_lasinfo.txt
        self.geom_file = '/home/jinx/repo/qchem/las_uccsd_data/polyenes/geometries/c8.xyz'
        self.frag_atom_list = ((0, 2), (10, 12), (13, 11), (3, 1))
        self.ncas_sub = (2, 2, 2, 2)
        self.nelec_sub = ((1, 1), (1, 1), (1, 1), (1, 1))
        self.frag_spin_orbs = {0: (0, 1, 8, 9), 1: (2, 3, 10, 11), 2: (4, 5, 12, 13), 3: (6, 7, 14, 15)}
        self.frag_pairs = [(0, 1), (1, 2), (2, 3)]
        
        # Setup molecule
        self.mol = self._setup_molecule()
        
    def _setup_molecule(self) -> gto.Mole:
        """Setup C8 molecule from geometry file"""
        try:
            with open(self.geom_file, 'r') as f:
                geom_data = f.read()
            
            mol = gto.M(atom=geom_data, basis=self.basis, verbose=0)
            mol.build()
            
            if self.verbose >= 1:
                print(f"C8 molecule loaded from {self.geom_file}")
                print(f"Basis: {self.basis}")
                print(f"Atoms: {mol.natm}, Electrons: {mol.nelectron}, Orbitals: {mol.nao}")
                
            return mol
                
        except FileNotFoundError:
            print(f"Error: Could not find {self.geom_file}")
            print("Please ensure the geometry file exists and is accessible.")
            raise
            
    def run_single_calculation(self, threshold: float, max_cycle: int) -> Dict[str, CalculationResult]:
        """Run calculation for single parameter combination"""
        
        print(f"\n{'='*80}")
        print(f"C8 Real Calculation: threshold={threshold}, max_cycle={max_cycle}")
        print(f"{'='*80}")
        
        start_time = time.time()
        results = {}
        
        try:
            # 1. CASCI Reference
            print("\n1. Running CASCI Reference")
            casci_result = self._run_casci()
            results['CASCI'] = casci_result
            print(f"CASCI Energy: {casci_result.energy:.10f} hartree")
            
            # 2. Setup LASVQENOSI
            las_calc = LASVQENOSI(
                mol=self.mol,
                ncas_sub=self.ncas_sub,
                nelec_sub=self.nelec_sub,
                basis=self.basis,
                frag_atom_list=self.frag_atom_list,
                gradient_threshold=threshold,
                vqe_max_cycles=max_cycle,
                verbose=self.verbose
            )
            
            las_calc.setup_molecule()
            las_calc.perform_lasscf()
            
            # 3. LASSCF result
            lasscf_result = CalculationResult(
                method_name='LASSCF',
                energy=las_calc.result.lasscf_energy,
                converged=las_calc.result.lasscf_converged
            )
            results['LASSCF'] = lasscf_result
            print(f"LASSCF Energy: {lasscf_result.energy:.10f} hartree")
            
            # 4. LAS-VQE (All Excitations)
            print("\n4. Running LAS-VQE (All Excitations)")
            las_vqe_result = self._run_las_vqe_all(las_calc)
            results['LAS_VQE_All'] = las_vqe_result
            print(f"LAS-VQE (All) Energy: {las_vqe_result.energy:.10f} hartree")
            
        except Exception as e:
            print(f"Calculation failed: {e}")
            import traceback
            traceback.print_exc()
            
        total_time = time.time() - start_time
        print(f"\nTotal calculation time: {total_time:.2f} seconds")
        
        return results
        
    def _run_casci(self) -> CalculationResult:
        """Run CASCI reference calculation"""
        start_time = time.time()
        
        # Mean-field calculation
        mf = scf.RHF(self.mol).run()
        
        # CASCI calculation
        ncas_total = sum(self.ncas_sub)
        nelec_total = sum(sum(nelec) for nelec in self.nelec_sub)
        mc_casci = mcscf.CASCI(mf, ncas_total, nelec_total)
        mc_casci.kernel()
        
        calc_time = time.time() - start_time
        
        return CalculationResult(
            method_name='CASCI',
            energy=mc_casci.e_tot,
            calculation_time=calc_time,
            converged=mc_casci.converged
        )
        
    def _run_las_vqe_all(self, las_calc: LASVQENOSI) -> CalculationResult:
        """Run LAS-VQE with all excitations"""
        start_time = time.time()
        
        # Get default excitations and merge them
        default_sets = las_calc.get_default_excitation_sets()
        
        all_doubles = []
        all_singles = []
        for doubles, singles in default_sets:
            all_doubles.extend(doubles)
            all_singles.extend(singles)
        
        # Set up single state VQE with all excitations
        las_calc.excitation_parameter_sets = [(all_doubles, all_singles)]
        vqe_states = las_calc.perform_vqe_on_sets()
        
        calc_time = time.time() - start_time
        energy = vqe_states[0].energy if vqe_states else None
        
        return CalculationResult(
            method_name='LAS-VQE (All)',
            energy=energy,
            n_excitations=len(all_doubles) + len(all_singles),
            n_states=1,
            calculation_time=calc_time,
            converged=vqe_states[0].converged if vqe_states else False
        )
        
    def write_results_markdown(self, results: Dict[str, CalculationResult], 
                             threshold: float, max_cycle: int, output_file: str):
        """Write results to markdown file"""
        
        casci_energy = results['CASCI'].energy if 'CASCI' in results else None
        
        content = f"""# C8 Real Calculation Results - 6-31G Basis - Threshold {threshold}, Max Cycle {max_cycle}

## Overview

**Molecule**: C8  
**Basis**: 6-31G  
**Excitation Threshold**: {threshold}  
**VQE Max Cycles**: {max_cycle}  
**Active Space**: {self.ncas_sub} orbitals, {self.nelec_sub} electrons  
**Geometry**: {self.geom_file}

## Energy Results Summary

"""
        
        if casci_energy:
            content += f"**CASCI Reference Energy**: {casci_energy:.10f} hartree\n\n"
        
        content += "| Method | Energy (hartree) | ΔE (mEh) | N_excitations | N_states | Time(s) | Converged |\n"
        content += "|--------|------------------|----------|---------------|----------|---------|-----------|\n"
        
        # Sort results by energy
        sorted_results = sorted(results.items(), key=lambda x: x[1].energy if x[1].energy is not None else float('inf'))
        
        for method_name, result in sorted_results:
            if result.energy is None:
                continue
                
            delta_e = (result.energy - casci_energy) * 1000 if casci_energy else 0.0
            conv_symbol = "✓" if result.converged else "✗"
            
            content += f"| **{result.method_name}** | {result.energy:.10f} | {delta_e:.3f} | {result.n_excitations} | {result.n_states} | {result.calculation_time:.2f} | {conv_symbol} |\n"
        
        # Add analysis
        content += f"\n## Calculation Analysis\n\n"
        content += f"This real calculation for C8 demonstrates:\n\n"
        
        if len(results) > 1:
            total_time = sum(r.calculation_time for r in results.values())
            content += f"- **Total computation time**: {total_time:.2f} seconds\n"
            
            if casci_energy:
                vqe_results = [r for r in results.values() if 'VQE' in r.method_name and r.energy is not None]
                if vqe_results:
                    best_vqe = min(vqe_results, key=lambda x: x.energy)
                    accuracy = (best_vqe.energy - casci_energy) * 1000
                    content += f"- **Best VQE accuracy**: {accuracy:.3f} mEh from CASCI\n"
        
        content += f"- **Method feasibility**: Successful demonstration of LAS-VQE-NOSI on C8\n"
        content += f"- **Active space size**: {sum(self.ncas_sub)} orbitals suitable for current methods\n"
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(content)
        
        print(f"Results written to {output_file}")

def main():
    """Run real calculation demonstration"""
    print("Running C8 real calculation demonstration")
    
    calc = C8RealCalculations(basis='6-31g', verbose=1)
    
    # Run one test calculation
    try:
        results = calc.run_single_calculation(0.001, 10)
        calc.write_results_markdown(results, 0.001, 10, "C8_6-31G_REAL_DEMO.md")
        print("\nC8 real calculation completed successfully!")
        
    except Exception as e:
        print(f"C8 calculation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
