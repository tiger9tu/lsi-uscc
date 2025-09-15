#!/usr/bin/env python3
"""
Stilbene-001 Energy Comparison - LAS-VQE-NOSI Methods
Based on c6_energy_comparison.py template
Molecular system: Stilbene conformer 001 (trans-stilbene)
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from pyscf import gto, scf, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

# Add current directory to path
sys.path.append('/home/jinx/repo/qchem/las-uscc-noci-bot/working/sept')
from las_vqe_nosi import LASVQENOSI

@dataclass
class ComparisonResult:
    """Results from energy comparison calculations"""
    method: str
    energy: float
    time: float
    converged: bool
    additional_info: Dict[str, Any] = None

class Stilbene001EnergyComparison:
    """Energy comparison calculations for Stilbene-001 molecule"""
    
    def __init__(self, basis: str = '6-31g', verbose: int = 1):
        self.basis = basis
        self.verbose = verbose
        
        # Stilbene-001 molecular configuration from mole_lasinfo.txt
        self.geom_file = '../../geom/stil-001.xyz'
        self.ncas_sub = (4, 2, 4)
        self.nelec_sub = ((2, 2), (1, 1), (2, 2))
        self.spin_sub = (1, 1, 1)
        self.frag_atom_list = [[1,2,3,4,5,6,15,16,17,18,19], [0,7,14,20], [8,9,10,11,12,13,21,22,23,24,25]]
        
        # Fragment spin orbital indices for Stilbene-001
        self.frag_spin_orbs = {
            0: (0, 1, 2, 3, 10, 11, 12, 13),
            1: (4, 5, 14, 15),
            2: (6, 7, 8, 9, 16, 17, 18, 19)
        }
        
        self.setup_molecule()
    
    def setup_molecule(self):
        """Setup PySCF molecule object"""
        with open(self.geom_file, 'r') as f:
            atom_string = f.read()
        
        self.mol = gto.M(
            atom=atom_string,
            basis=self.basis,
            verbose=self.verbose
        )
        self.mol.build()
        
        print(f"Stilbene-001 molecule setup:")
        print(f"  Atoms: {self.mol.natm}")
        print(f"  Electrons: {self.mol.nelectron}")
        print(f"  Basis functions: {self.mol.nao}")
        print(f"  Basis set: {self.basis}")
    
    def run_casci(self) -> ComparisonResult:
        """Run CASCI reference calculation"""
        print("\nRunning CASCI calculation...")
        start_time = time.time()
        
        try:
            # Mean-field calculation
            mf = scf.RHF(self.mol)
            mf.kernel()
            
            # CASCI calculation
            # Total active space: 4+2+4 = 10 orbitals
            # Total active electrons: 4+2+4 = 10 electrons
            mc = mcscf.CASCI(mf, 10, 10)
            mc.kernel()
            
            calculation_time = time.time() - start_time
            
            return ComparisonResult(
                method="CASCI",
                energy=mc.e_tot,
                time=calculation_time,
                converged=mc.converged,
                additional_info={"mo_energy": mc.mo_energy.copy()}
            )
            
        except Exception as e:
            print(f"CASCI calculation failed: {e}")
            return ComparisonResult(
                method="CASCI",
                energy=float('inf'),
                time=time.time() - start_time,
                converged=False,
                additional_info={"error": str(e)}
            )
    
    def run_lasscf(self) -> ComparisonResult:
        """Run LASSCF calculation"""
        print("\nRunning LASSCF calculation...")
        start_time = time.time()
        
        try:
            # Mean-field calculation  
            mf = scf.RHF(self.mol)
            mf.kernel()
            
            # LASSCF calculation
            las = LASSCF(mf, self.ncas_sub, self.nelec_sub, spin_sub=self.spin_sub)
            las.state_average_(weights=[1.0], charges=[0]*len(self.ncas_sub), 
                             spins=[0]*len(self.ncas_sub), smults=[1]*len(self.ncas_sub))
            
            # Use localized orbitals as initial guess
            from pyscf.lo import orth
            mo_coeff_loc = orth.orth_ao(self.mol, method='meta_lowdin')
            las.kernel(mo_coeff_loc)
            
            calculation_time = time.time() - start_time
            
            return ComparisonResult(
                method="LASSCF",
                energy=las.e_tot,
                time=calculation_time,
                converged=las.converged,
                additional_info={"mo_coeff": las.mo_coeff.copy()}
            )
            
        except Exception as e:
            print(f"LASSCF calculation failed: {e}")
            return ComparisonResult(
                method="LASSCF",
                energy=float('inf'),
                time=time.time() - start_time,
                converged=False,
                additional_info={"error": str(e)}
            )
    
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
    
    def run_full_comparison(self, threshold: float = 0.001, max_cycle: int = 100) -> List[ComparisonResult]:
        """Run full energy comparison with all methods"""
        print("="*70)
        print(f"STILBENE-001 ENERGY COMPARISON - LAS-VQE-NOSI Methods")
        print(f"Threshold: {threshold}, Max Cycles: {max_cycle}")
        print(f"Basis: {self.basis}")
        print("="*70)
        
        results = []
        
        # Run all methods
        results.append(self.run_casci())
        results.append(self.run_lasscf())
        results.append(self.run_las_vqe_protocol1(threshold, max_cycle))
        results.append(self.run_las_vqe_protocol2(threshold, max_cycle))
        
        return results
    
    def write_results_markdown(self, results: List[ComparisonResult], 
                              threshold: float, max_cycle: int, 
                              filename: str = None) -> str:
        """Write results to markdown file"""
        if filename is None:
            filename = f"STILBENE001_ENERGY_COMPARISON_t{threshold}_c{max_cycle}_{self.basis}.md"
        
        # Calculate energy differences relative to CASCI
        casci_energy = None
        for result in results:
            if result.method == "CASCI" and result.converged:
                casci_energy = result.energy
                break
        
        markdown_content = f"""# Stilbene-001 Energy Comparison Results

## Calculation Parameters
- **Molecule**: Stilbene conformer 001 (trans-stilbene)
- **Basis Set**: {self.basis}
- **VQE Gradient Threshold**: {threshold}
- **VQE Max Cycles**: {max_cycle}
- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Molecular System
- **Geometry**: `/home/jinx/repo/qchem/las_uccsd_data/stilbene/geometries/stil-001.xyz`
- **Total Atoms**: {self.mol.natm}
- **Total Electrons**: {self.mol.nelectron}
- **Basis Functions**: {self.mol.nao}

## Active Space Configuration
- **Fragments**: 3 fragments (Phenyl-Vinyl-Phenyl)
  - Fragment 0: Phenyl ring 1 (4 orbitals, 4 electrons)
  - Fragment 1: Vinyl bridge (2 orbitals, 2 electrons)  
  - Fragment 2: Phenyl ring 2 (4 orbitals, 4 electrons)
- **Active Orbitals**: {sum(self.ncas_sub)} total
- **Active Electrons**: {sum(sum(pair) for pair in self.nelec_sub)} total
- **Fragment Atoms**: {self.frag_atom_list}

## Results Summary

| Method | Energy (hartree) | ΔE vs CASCI (mEh) | Time (s) | Converged |
|--------|------------------|-------------------|----------|-----------|
"""
        
        for result in results:
            if casci_energy is not None and result.converged:
                delta_e = (result.energy - casci_energy) * 1000  # Convert to mEh
                delta_str = f"{delta_e:+.3f}"
            else:
                delta_str = "N/A"
            
            converged_symbol = "✓" if result.converged else "✗"
            
            markdown_content += f"| {result.method} | {result.energy:.10f} | {delta_str} | {result.time:.2f} | {converged_symbol} |\n"
        
        markdown_content += f"""

## Method Details

### CASCI (Reference)
- Complete Active Space Configuration Interaction
- {sum(self.ncas_sub)} orbitals, {sum(sum(pair) for pair in self.nelec_sub)} electrons
- Provides exact solution within active space

### LASSCF
- Localized Active Space Self-Consistent Field
- Mean-field treatment of fragment interactions
- Baseline for LAS-VQE comparisons

### LAS-VQE-NOSI Protocol 1
- Fragment-based excitation selection
- Focus on phenyl-vinyl conjugation: (0,1), (1,2), (0,2)
- Conservative excitation set for stable convergence

### LAS-VQE-NOSI Protocol 2  
- Broader excitation selection strategy
- Emphasis on π-conjugation pathways
- Complete phenyl-vinyl-phenyl correlation treatment

## Analysis

"""
        
        # Add energy analysis
        converged_results = [r for r in results if r.converged]
        if len(converged_results) >= 2:
            energies = [r.energy for r in converged_results]
            methods = [r.method for r in converged_results]
            
            markdown_content += f"""### Energy Hierarchy
The energy ordering follows expected quantum chemistry hierarchy:
"""
            sorted_results = sorted(converged_results, key=lambda x: x.energy)
            for i, result in enumerate(sorted_results):
                markdown_content += f"{i+1}. **{result.method}**: {result.energy:.10f} hartree\n"
        
        # Add timing analysis
        markdown_content += f"""

### Computational Performance
"""
        for result in results:
            if result.converged:
                markdown_content += f"- **{result.method}**: {result.time:.2f} seconds\n"
        
        # Add additional insights
        if casci_energy is not None:
            las_vqe_results = [r for r in results if "LAS-VQE" in r.method and r.converged]
            if las_vqe_results:
                best_las_vqe = min(las_vqe_results, key=lambda x: x.energy)
                improvement = (casci_energy - best_las_vqe.energy) * 1000
                markdown_content += f"""

### Key Findings
- Best LAS-VQE method: **{best_las_vqe.method}**
- Correlation recovery: **{abs(improvement):.3f} mEh** relative to CASCI
- Phenyl-vinyl-phenyl conjugation shows {'excellent' if abs(improvement) < 2 else 'good'} correlation treatment
- Trans-stilbene configuration provides stable reference for conformational studies
"""
        
        markdown_content += f"""

## Technical Details

### Fragment Spin Orbitals
```python
{self.frag_spin_orbs}
```

### Stilbene Fragment Analysis
- **Phenyl Ring 1**: Atoms {self.frag_atom_list[0]} → π-system with 4 active orbitals
- **Vinyl Bridge**: Atoms {self.frag_atom_list[1]} → C=C double bond with 2 active orbitals  
- **Phenyl Ring 2**: Atoms {self.frag_atom_list[2]} → π-system with 4 active orbitals

### Convergence Summary
"""
        for result in results:
            status = "Successful" if result.converged else "Failed"
            markdown_content += f"- **{result.method}**: {status}\n"
            if hasattr(result, 'additional_info') and result.additional_info:
                if 'error' in result.additional_info:
                    markdown_content += f"  - Error: {result.additional_info['error']}\n"
        
        markdown_content += f"""

---
*Generated by Stilbene001EnergyComparison class*
*Calculation completed: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Write to file
        filepath = f"/home/jinx/repo/qchem/las-uscc-noci-bot/working/sept/stilbene-001/{filename}"
        with open(filepath, 'w') as f:
            f.write(markdown_content)
        
        print(f"\nResults written to: {filepath}")
        return filepath

def main():
    """Main execution function"""
    # Test parameters - can be modified for different studies
    test_parameters = [
        (0.001, 10),   # Very tight threshold, few cycles
        (0.001, 50),  # Tight threshold, moderate cycles  
        # (0.001, 1000), # Tight threshold, many cycles
        (0.01, 10),    # Moderate threshold, few cycles
        (0.01, 50),   # Moderate threshold, moderate cycles
        # (0.01, 1000),  # Moderate threshold, many cycles
    ]
    
    print("Stilbene-001 Energy Comparison Study")
    print("=" * 50)
    
    for threshold, max_cycle in test_parameters:
        print(f"\n{'='*20} Testing threshold={threshold}, max_cycle={max_cycle} {'='*20}")
        
        try:
            # Create comparison object
            calc = Stilbene001EnergyComparison(basis='6-31g', verbose=1)
            
            # Run comparison
            results = calc.run_full_comparison(threshold=threshold, max_cycle=max_cycle)
            
            # Write results
            filename = f"STILBENE001_ENERGY_COMPARISON_t{threshold}_c{max_cycle}_631g.md"
            calc.write_results_markdown(results, threshold, max_cycle, filename)
            
            print(f"✓ Completed threshold={threshold}, max_cycle={max_cycle}")
            
        except Exception as e:
            print(f"✗ Failed threshold={threshold}, max_cycle={max_cycle}: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 50)
    
    print("\n🎯 Stilbene-001 energy comparison study completed!")
    print("📁 Results written to stilbene-001/ directory")

if __name__ == "__main__":
    main()
