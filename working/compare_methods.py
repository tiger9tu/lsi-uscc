#!/usr/bin/env python3
"""
Comparison of LAS-VQE-SI, LASSI, and CASCI methods

This script performs ground state energy calculations using three different methods:
1. LAS-VQE-SI (Localized Active Space - Variational Quantum Eigensolver - State Interaction)
2. LASSI (Localized Active Space - State Interaction)  
3. CASCI (Complete Active Space Configuration Interaction)

For H4 and H6 molecular systems.
"""

import numpy as np
from pyscf import gto, scf, mcscf, lib
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSI
from mrh.my_pyscf.fci import csf_solver

# Import the VQE class from the working directory
from lassi_vqe_class import LASSI_VQE

# Molecular configurations (same as before)
H4xyz = ''' H      0.000000000000   0.000000000000   0.000000000000
H      1.000000000000   0.000000000000   0.000000000000
H      0.273746762116   2.195450598147   0.100000000000
H      1.232912762116   1.895450598147  -0.100000000000
'''

H6xyz = ''' H      0.000000000000   0.000000000000   0.000000000000
H      1.000000000000   0.000000000000   0.000000000000
H      0.273746762116   2.195450598147   0.100000000000
H      1.232912762116   1.895450598147  -0.100000000000
H      0.507178110854   4.193780995243   0.049334760036
H      1.506140937609   3.988021397347  -0.049334760036
'''

h4_config = {
    'name': 'H4',
    'xyz': H4xyz,
    'basis': 'sto-3g',
    'ncas': [2, 2],
    'nelecas': [2, 2],
    'spinsub': [1, 1],
    'frag_atom_list': ((0, 1), (2, 3)),
    'total_ncas': 4,
    'total_nelecas': 4
}

h6_config = {
    'name': 'H6', 
    'xyz': H6xyz,
    'basis': 'sto-3g',
    'ncas': [2, 2, 2],
    'nelecas': [2, 2, 2],
    'spinsub': [1, 1, 1],
    'frag_atom_list': ((0, 1), (2, 3), (4, 5)),
    'total_ncas': 6,
    'total_nelecas': 6
}

def setup_molecule(config):
    """Setup molecule and HF reference"""
    mol = gto.M(
        atom=config['xyz'],
        basis=config['basis'],
        verbose=0,
        output=None
    )
    mf = scf.RHF(mol).run()
    return mol, mf

def run_casci(mol, mf, config):
    """Run CASCI calculation"""
    print(f"  Running CASCI({config['total_ncas']},{config['total_nelecas']})...")
    
    # Use CSF solver for proper spin treatment
    mc = mcscf.CASCI(mf, config['total_ncas'], config['total_nelecas'])
    mc.fcisolver = csf_solver(mol, smult=1)  # Singlet ground state
    mc.kernel()
    
    if not mc.converged:
        print(f"  Warning: CASCI not converged for {config['name']}")
    
    return mc.e_tot

def run_lasscf_setup(mol, mf, config):
    """Setup LASSCF calculation (common for LASSI and LAS-VQE-SI)"""
    print(f"  Setting up LASSCF fragmentation...")
    
    # Setup LASSCF
    las = LASSCF(mf, config['ncas'], config['nelecas'], spin_sub=config['spinsub'])
    
    # Localize initial guess
    mo_loc = las.localize_init_guess(config['frag_atom_list'], mf.mo_coeff)
    las.kernel(mo_loc)
    
    if not las.converged:
        print(f"  Warning: LASSCF not converged for {config['name']}")
    
    print(f"  LASSCF energy: {las.e_tot:.10f} hartree")
    return las

def run_lassi(las, config):
    """Run LASSI calculation"""
    print(f"  Running LASSI calculation...")
    
    # Setup state averaging (simple ground state for fair comparison)
    las_sa = las.state_average([1.0], spins=[[1]*len(config['ncas'])], 
                              smults=[[2]*len(config['ncas'])], 
                              charges=[[0]*len(config['ncas'])])
    las_sa.lasci()
    
    # Create and run LASSI
    lsi = LASSI(las_sa)
    e_roots, si = lsi.kernel()
    
    return e_roots[0]  # Ground state energy

def run_las_vqe_si(mol, mf, config):
    """Run LAS-VQE-SI calculation"""
    print(f"  Running LAS-VQE-SI calculation...")
    
    try:
        # Convert nelecas format for LASSI_VQE
        nelec_sub = []
        for i, nfrags in enumerate(config['ncas']):
            # For each fragment, distribute electrons as (alpha, beta)
            # For simplicity, assume closed shell: nelec/2 alpha, nelec/2 beta
            nelec_frag = config['nelecas'][i]
            if isinstance(nelec_frag, int):
                # If total electrons given, distribute equally
                nelec_sub.append((nelec_frag // 2, nelec_frag // 2))
            else:
                # If already in (alpha, beta) format
                nelec_sub.append(nelec_frag)
        
        # Setup LAS-VQE-SI instance
        vqe_si = LASSI_VQE(mol, tuple(config['ncas']), tuple(nelec_sub), 
                          basis=config['basis'], verbose=0)
        
        # Run the VQE optimization
        results_dict = vqe_si.run_full_calculation(vqe_max_cycle=10, vqe_conv_tol=1e-4)
        vqe_energy = results_dict.get('vqe_si_energy')
        
        return vqe_energy
        
    except Exception as e:
        print(f"  Error in LAS-VQE-SI: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_methods_single_molecule(config):
    """Compare all three methods for a single molecule"""
    print(f"\n{'='*60}")
    print(f"COMPARING METHODS FOR {config['name']}")
    print(f"{'='*60}")
    
    # Setup molecule
    mol, mf = setup_molecule(config)
    print(f"HF energy: {mf.e_tot:.10f} hartree")
    
    results = {}
    
    # 1. CASCI calculation
    print(f"\n1. CASCI Method:")
    try:
        casci_energy = run_casci(mol, mf, config)
        results['CASCI'] = casci_energy
        print(f"  ✓ CASCI energy: {casci_energy:.10f} hartree")
    except Exception as e:
        print(f"  ✗ CASCI failed: {e}")
        results['CASCI'] = None
    
    # 2. Setup LASSCF (common for LASSI and LAS-VQE-SI)
    print(f"\n2. LASSCF Setup:")
    try:
        las = run_lasscf_setup(mol, mf, config)
        
        # 3. LASSI calculation
        print(f"\n3. LASSI Method:")
        try:
            lassi_energy = run_lassi(las, config)
            results['LASSI'] = lassi_energy
            print(f"  ✓ LASSI energy: {lassi_energy:.10f} hartree")
        except Exception as e:
            print(f"  ✗ LASSI failed: {e}")
            results['LASSI'] = None
        
        # 4. LAS-VQE-SI calculation
        print(f"\n4. LAS-VQE-SI Method:")
        try:
            vqe_si_energy = run_las_vqe_si(mol, mf, config)
            if vqe_si_energy is not None:
                results['LAS-VQE-SI'] = vqe_si_energy
                print(f"  ✓ LAS-VQE-SI energy: {vqe_si_energy:.10f} hartree")
            else:
                results['LAS-VQE-SI'] = None
        except Exception as e:
            print(f"  ✗ LAS-VQE-SI failed: {e}")
            results['LAS-VQE-SI'] = None
            
    except Exception as e:
        print(f"  ✗ LASSCF setup failed: {e}")
        results['LASSI'] = None
        results['LAS-VQE-SI'] = None
    
    return results

def print_comparison_table(h4_results, h6_results):
    """Print comparison table of all results"""
    print(f"\n{'='*80}")
    print("GROUND STATE ENERGY COMPARISON")
    print(f"{'='*80}")
    
    # Table header
    print(f"{'Method':<15} {'H4 Energy (hartree)':<20} {'H6 Energy (hartree)':<20} {'Δ(H6-H4)':<15}")
    print(f"{'-'*15} {'-'*20} {'-'*20} {'-'*15}")
    
    methods = ['CASCI', 'LASSI', 'LAS-VQE-SI']
    
    for method in methods:
        h4_e = h4_results.get(method)
        h6_e = h6_results.get(method)
        
        h4_str = f"{h4_e:.10f}" if h4_e is not None else "Failed"
        h6_str = f"{h6_e:.10f}" if h6_e is not None else "Failed"
        
        if h4_e is not None and h6_e is not None:
            delta_str = f"{h6_e - h4_e:.10f}"
        else:
            delta_str = "N/A"
        
        print(f"{method:<15} {h4_str:<20} {h6_str:<20} {delta_str:<15}")
    
    # Energy differences between methods
    if all(h4_results.get(m) is not None for m in methods):
        print(f"\nH4 Energy Differences (hartree):")
        casci_h4 = h4_results['CASCI']
        print(f"  LASSI - CASCI:     {h4_results['LASSI'] - casci_h4:.10f}")
        print(f"  LAS-VQE-SI - CASCI: {h4_results['LAS-VQE-SI'] - casci_h4:.10f}")
        print(f"  LAS-VQE-SI - LASSI: {h4_results['LAS-VQE-SI'] - h4_results['LASSI']:.10f}")
    
    if all(h6_results.get(m) is not None for m in methods):
        print(f"\nH6 Energy Differences (hartree):")
        casci_h6 = h6_results['CASCI']
        print(f"  LASSI - CASCI:     {h6_results['LASSI'] - casci_h6:.10f}")
        print(f"  LAS-VQE-SI - CASCI: {h6_results['LAS-VQE-SI'] - casci_h6:.10f}")
        print(f"  LAS-VQE-SI - LASSI: {h6_results['LAS-VQE-SI'] - h6_results['LASSI']:.10f}")

def main():
    """Main comparison function"""
    print("GROUND STATE ENERGY COMPARISON: LAS-VQE-SI vs LASSI vs CASCI")
    print("="*70)
    
    # Run calculations for both molecules
    h4_results = compare_methods_single_molecule(h4_config)
    h6_results = compare_methods_single_molecule(h6_config)
    
    # Print comparison table
    print_comparison_table(h4_results, h6_results)
    
    return h4_results, h6_results

if __name__ == "__main__":
    results = main()