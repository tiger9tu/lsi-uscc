#!/usr/bin/env python3
"""
Example: Complete LASSI calculation for a molecule from test_nop_org.py

This demonstrates how to:
1. Set up the molecular system and LASSCF calculation using LAS parameters
2. Define state averaging parameters (weights, spins, smults, charges)  
3. Construct and run LASSI diagonalization
4. Analyze the results
"""

import numpy as np
from lassi_instances import construct_lassi_instance, h4_sto3g, h6_sto3g

def run_full_lassi_example(mol_config, run_lassi=True):
    """Run complete LASSI calculation and analysis"""
    
    print(f"\n{'='*60}")
    print(f"COMPLETE LASSI CALCULATION: {mol_config['name']}")
    print(f"{'='*60}")
    
    # Step 1: Construct LASSI instance
    print("\n1. Constructing LASSI instance...")
    lsi, las_sa = construct_lassi_instance(mol_config, verbose=0)
    
    print(f"   ✓ LASSCF energy: {las_sa.e_tot:.10f} hartree")
    print(f"   ✓ Fragment setup: {len(mol_config['ncas'])} fragments")
    print(f"   ✓ Active orbitals: {mol_config['ncas']} per fragment")
    print(f"   ✓ Active electrons: {mol_config['nelecas']} per fragment") 
    print(f"   ✓ Number of states: {las_sa.nroots}")
    
    if not run_lassi:
        print("\n   (Skipping LASSI diagonalization)")
        return lsi, las_sa
    
    # Step 2: Run LASSI diagonalization
    print("\n2. Running LASSI diagonalization...")
    try:
        e_roots, si = lsi.kernel()
        print(f"   ✓ LASSI completed successfully!")
        print(f"   ✓ Number of eigenvalues: {len(e_roots)}")
    except Exception as e:
        print(f"   ✗ LASSI diagonalization failed: {e}")
        return lsi, las_sa
    
    # Step 3: Analyze results
    print("\n3. Analysis of LASSI results:")
    print(f"   Ground state energy: {e_roots[0]:.10f} hartree")
    
    if len(e_roots) > 1:
        excitation_energies = (e_roots[1:] - e_roots[0]) * 27.2114  # Convert to eV
        print(f"   Excitation energies (eV):")
        for i, exc_e in enumerate(excitation_energies[:5]):  # Show first 5
            print(f"     State {i+1}: {exc_e:.6f} eV")
    
    # Show spin and electron information if available
    if hasattr(si, 's2'):
        print(f"   <S²> expectation values: {si.s2[:5]}")  # Show first 5
    if hasattr(si, 'nelec'):
        print(f"   Electron configurations: {si.nelec[:5]}")  # Show first 5
    
    # Show state compositions
    print(f"   SI vector shape: {si.shape}")
    print(f"   Largest coefficients in ground state:")
    ground_state_coeffs = si[:, 0]
    largest_indices = np.argsort(np.abs(ground_state_coeffs))[-3:][::-1]
    for idx in largest_indices:
        print(f"     State {idx}: {ground_state_coeffs[idx]:.6f}")
    
    return lsi, las_sa, e_roots, si

def main():
    """Run examples for different molecular systems"""
    
    # Example 1: H4 - Simple 2-fragment system
    print("Running LASSI example for H4 (2 fragments)...")
    lsi_h4, las_h4, e_h4, si_h4 = run_full_lassi_example(h4_sto3g)
    
    # Example 2: H6 - 3-fragment system  
    print("\n\nRunning LASSI example for H6 (3 fragments)...")
    lsi_h6, las_h6, e_h6, si_h6 = run_full_lassi_example(h6_sto3g)
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"H4 ground state energy: {e_h4[0]:.10f} hartree")
    print(f"H6 ground state energy: {e_h6[0]:.10f} hartree")
    
    if len(e_h4) > 1:
        h4_gap = (e_h4[1] - e_h4[0]) * 27.2114
        print(f"H4 first excitation:     {h4_gap:.6f} eV")
    
    if len(e_h6) > 1:
        h6_gap = (e_h6[1] - e_h6[0]) * 27.2114  
        print(f"H6 first excitation:     {h6_gap:.6f} eV")
    
    print(f"\nLASSI calculations completed successfully!")
    return {
        'h4': {'lsi': lsi_h4, 'las': las_h4, 'e_roots': e_h4, 'si': si_h4},
        'h6': {'lsi': lsi_h6, 'las': las_h6, 'e_roots': e_h6, 'si': si_h6}
    }

if __name__ == "__main__":
    results = main()