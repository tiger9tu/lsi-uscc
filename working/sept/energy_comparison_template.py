#!/usr/bin/env python3
"""
Energy Comparison Template - LAS-VQE-NOSI Methods
Template for molecular energy comparison studies
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from pyscf import gto, scf, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from las_vqe_nosi import LASVQENOSI

@dataclass
class ComparisonResult:
    """Results from energy comparison calculations"""
    method: str
    energy: float
    time: float
    converged: bool
    additional_info: Dict[str, Any] = None

class MolecularEnergyComparison:
    """Base class for molecular energy comparison calculations"""
    
    def __init__(self, molecule_name: str, geom_filename: str, basis: str = '6-31g', verbose: int = 1):
        self.molecule_name = molecule_name
        self.basis = basis
        self.verbose = verbose
        
        # Try multiple possible paths for geometry file
        possible_paths = [
            geom_filename,  # Direct path if provided
            f'../geom/{geom_filename}',  # Relative to working directory
            f'../../geom/{geom_filename}',  # Common pattern
            f'/home/jinx/repo/qchem/las_uccsd_data/polyenes/geometries/{geom_filename}',  # C chains
            f'/home/jinx/repo/qchem/las_uccsd_data/stilbene/geometries/{geom_filename}',  # Stilbene
        ]
        
        self.geom_file = None
        for path in possible_paths:
            if os.path.exists(path):
                self.geom_file = path
                break
        
        if self.geom_file is None:
            raise FileNotFoundError(f"Could not find geometry file {geom_filename} in any of the expected locations")
        
        # These should be set by subclasses
        self.ncas_sub = None
        self.nelec_sub = None
        self.spin_sub = None
        self.frag_atom_list = None
        self.frag_spin_orbs = None
        
        # Setup molecule after subclass initialization
        self.mol = None
    
    def setup_molecule(self):
        """Setup PySCF molecule object"""
        if self.geom_file is None:
            raise ValueError("Geometry file not found")
            
        with open(self.geom_file, 'r') as f:
            atom_string = f.read().strip()
        
        self.mol = gto.M(
            atom=atom_string,
            basis=self.basis,
            verbose=self.verbose if self.verbose > 0 else 0
        )
        self.mol.build()
        
        if self.verbose >= 1:
            print(f"{self.molecule_name} molecule setup:")
            print(f"  Geometry: {self.geom_file}")
            print(f"  Atoms: {self.mol.natm}")
            print(f"  Electrons: {self.mol.nelectron}")
            print(f"  Basis functions: {self.mol.nao}")
            print(f"  Basis set: {self.basis}")
    
    def run_casci(self) -> ComparisonResult:
        """Run CASCI reference calculation"""
        if self.verbose >= 1:
            print("\nRunning CASCI calculation...")
        start_time = time.time()
        
        try:
            # Mean-field calculation
            mf = scf.RHF(self.mol)
            mf.verbose = 0
            mf.kernel()
            
            # CASCI calculation
            ncas_total = sum(self.ncas_sub)
            nelec_total = sum(sum(pair) for pair in self.nelec_sub)
            mc = mcscf.CASCI(mf, ncas_total, nelec_total)
            mc.verbose = 0
            mc.kernel()
            
            calculation_time = time.time() - start_time
            
            return ComparisonResult(
                method="CASCI",
                energy=mc.e_tot,
                time=calculation_time,
                converged=mc.converged,
                additional_info={"mo_energy": mc.mo_energy.copy() if hasattr(mc, 'mo_energy') else None}
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
        if self.verbose >= 1:
            print("\nRunning LASSCF calculation...")
        start_time = time.time()
        
        try:
            # Mean-field calculation  
            mf = scf.RHF(self.mol)
            mf.verbose = 0
            mf.kernel()
            
            # LASSCF calculation
            las = LASSCF(mf, self.ncas_sub, self.nelec_sub, spin_sub=self.spin_sub)
            las.state_average_(
                weights=[1.0], 
                charges=[0]*len(self.ncas_sub), 
                spins=[0]*len(self.ncas_sub), 
                smults=[1]*len(self.ncas_sub)
            )
            
            # Use localized orbitals as initial guess
            from pyscf.lo import orth
            mo_coeff_loc = orth.orth_ao(self.mol, method='meta_lowdin')
            las.verbose = 0
            las.kernel(mo_coeff_loc)
            
            calculation_time = time.time() - start_time
            
            return ComparisonResult(
                method="LASSCF",
                energy=las.e_tot,
                time=calculation_time,
                converged=las.converged,
                additional_info={"mo_coeff": las.mo_coeff.copy() if hasattr(las, 'mo_coeff') else None}
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
            # This method should be implemented by subclasses
            # as excitation generation is molecule-specific
            raise NotImplementedError("Subclass must implement run_las_vqe_protocol1")
            
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
            # This method should be implemented by subclasses
            # as excitation generation is molecule-specific
            raise NotImplementedError("Subclass must implement run_las_vqe_protocol2")
            
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
        print(f"{self.molecule_name.upper()} ENERGY COMPARISON - LAS-VQE-NOSI Methods")
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
            filename = f"{self.molecule_name.upper()}_ENERGY_COMPARISON_t{threshold}_c{max_cycle}_{self.basis}.md"
        
        # Calculate energy differences relative to CASCI
        casci_energy = None
        for result in results:
            if result.method == "CASCI" and result.converged:
                casci_energy = result.energy
                break
        
        markdown_content = f"""# {self.molecule_name} Energy Comparison Results

## Calculation Parameters
- **Molecule**: {self.molecule_name}
- **Basis Set**: {self.basis}
- **VQE Gradient Threshold**: {threshold}
- **VQE Max Cycles**: {max_cycle}
- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Molecular System
- **Geometry**: {self.geom_file}
- **Total Atoms**: {self.mol.natm}
- **Total Electrons**: {self.mol.nelectron}
- **Basis Functions**: {self.mol.nao}

## Active Space Configuration
- **Active Orbitals**: {sum(self.ncas_sub)} total ({self.ncas_sub})
- **Active Electrons**: {sum(sum(pair) for pair in self.nelec_sub)} total ({self.nelec_sub})
- **Fragment Atoms**: {self.frag_atom_list}

## Results Summary

| Method | Energy (hartree) | ΔE vs CASCI (mEh) | Time (s) | Converged |
|--------|------------------|-------------------|----------|-----------|
"""
        
        for result in results:
            if casci_energy is not None and result.converged and result.energy != float('inf'):
                delta_e = (result.energy - casci_energy) * 1000  # Convert to mEh
                delta_str = f"{delta_e:+.3f}"
            else:
                delta_str = "N/A"
            
            converged_symbol = "✓" if result.converged else "✗"
            energy_str = f"{result.energy:.10f}" if result.energy != float('inf') else "Failed"
            
            markdown_content += f"| {result.method} | {energy_str} | {delta_str} | {result.time:.2f} | {converged_symbol} |\n"
        
        markdown_content += f"""

## Analysis

### Energy Hierarchy
"""
        converged_results = [r for r in results if r.converged and r.energy != float('inf')]
        if len(converged_results) >= 2:
            sorted_results = sorted(converged_results, key=lambda x: x.energy)
            for i, result in enumerate(sorted_results):
                markdown_content += f"{i+1}. **{result.method}**: {result.energy:.10f} hartree\n"
        
        markdown_content += f"""

### Computational Performance
"""
        for result in results:
            if result.converged:
                markdown_content += f"- **{result.method}**: {result.time:.2f} seconds\n"
        
        if casci_energy is not None:
            las_vqe_results = [r for r in results if "LAS-VQE" in r.method and r.converged and r.energy != float('inf')]
            if las_vqe_results:
                best_las_vqe = min(las_vqe_results, key=lambda x: x.energy)
                improvement = (casci_energy - best_las_vqe.energy) * 1000
                markdown_content += f"""

### Key Findings
- Best LAS-VQE method: **{best_las_vqe.method}**
- Correlation recovery: **{abs(improvement):.3f} mEh** relative to CASCI
"""
        
        markdown_content += f"""

### Technical Details

#### Fragment Spin Orbitals
```python
{self.frag_spin_orbs}
```

#### Convergence Summary
"""
        for result in results:
            status = "Successful" if result.converged else "Failed"
            markdown_content += f"- **{result.method}**: {status}\n"
            if hasattr(result, 'additional_info') and result.additional_info and 'error' in result.additional_info:
                markdown_content += f"  - Error: {result.additional_info['error']}\n"
        
        markdown_content += f"""

---
*Generated by {self.__class__.__name__}*
*Calculation completed: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Write to file in current directory
        with open(filename, 'w') as f:
            f.write(markdown_content)
        
        print(f"\nResults written to: {filename}")
        return filename

def run_standard_test_suite(comparison_instance, molecule_name: str):
    """Run standard test parameter combinations"""
    test_parameters = [
        (0.01, 10),    # Fast test 
        (0.001, 50),   # Moderate test
    ]
    
    print(f"{molecule_name} Energy Comparison Study")
    print("=" * 50)
    
    for threshold, max_cycle in test_parameters:
        print(f"\n{'='*20} Testing threshold={threshold}, max_cycle={max_cycle} {'='*20}")
        
        try:
            # Run comparison
            results = comparison_instance.run_full_comparison(threshold=threshold, max_cycle=max_cycle)
            
            # Write results
            filename = f"{molecule_name.upper()}_ENERGY_COMPARISON_t{threshold}_c{max_cycle}_631g.md"
            comparison_instance.write_results_markdown(results, threshold, max_cycle, filename)
            
            print(f"✓ Completed threshold={threshold}, max_cycle={max_cycle}")
            
        except Exception as e:
            print(f"✗ Failed threshold={threshold}, max_cycle={max_cycle}: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 50)
    
    print(f"\n🎯 {molecule_name} energy comparison study completed!")