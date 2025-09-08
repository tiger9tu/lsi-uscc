#!/usr/bin/env python3
"""
Example usage of LASSIrq implementation

This script demonstrates how to use the updated LASSI_VQE class and LASSIrq instances
with automatic charge transfer state generation instead of manual state averaging.
"""

import numpy as np
from pyscf import gto, scf, lib
from lassi_vqe_class import LASSI_VQE
from lassi_instances_lassirq import construct_lassirq_instance, h4_sto3g, h6_sto3g
from c2h4n4_struct import structure as struct

def example_h4_lassirq():
    """Example 1: H4 molecule with LASSIrq"""
    print("=" * 60)
    print("EXAMPLE 1: H4 with LASSIrq")
    print("=" * 60)
    
    # Construct LASSIrq instance using the new approach
    lsi, las = construct_lassirq_instance(h4_sto3g, verbose=1)
    
    # Run LASSIrq calculation
    print("\nRunning LASSIrq kernel...")
    e_roots, si_rq = lsi.kernel()
    
    print(f"\nLASSIrq[1,1] results for H4:")
    print(f"Ground state energy: {e_roots[0]:.10f} hartree")
    print(f"All energies: {e_roots}")
    print(f"Ground state vector: {si_rq[:,0]}")
    
    return lsi, las

def example_c2h4n4_lassirq_vqe():
    """Example 2: C2H4N4 molecule with LASSI-VQE using LASSIrq"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: C2H4N4 with LASSIrq-VQE")
    print("=" * 60)
    
    # Initialize LASSI_VQE class with LASSIrq parameters (r=1, q=1)
    lassi_vqe = LASSI_VQE(
        mol=struct,  # C2H4N4 structure function
        ncas_sub=(3, 3),
        nelec_sub=((2, 1), (1, 2)),
        basis='6-31g',
        output_file='c2h4n4_lassirq_vqe.log',
        verbose=lib.logger.INFO,
        r=1,  # 1-electron charge transfer
        q=1   # 1-electron charge transfer
    )
    
    # Run the full LASSIrq-VQE calculation
    print("\nRunning LASSIrq-VQE calculation...")
    try:
        # Define fragment atom lists for C2H4N4
        frag_atom_list = (list(range(5)), list(range(5, 10)))
        
        final_energies, final_eigenvectors, lassirq_energies = lassi_vqe.run_full_calculation(
            frag_atom_list=frag_atom_list,
            vqe_max_cycle=3,  # Reduced for demo
            vqe_conv_tol=1e-6
        )
        
        print(f"\nFinal Results:")
        print(f"VQE-LASSIrq ground state energy: {final_energies[0]:.10f} hartree")
        print(f"LASSIrq[1,1] ground state energy: {lassirq_energies[0]:.10f} hartree")
        print(f"Energy difference: {final_energies[0] - lassirq_energies[0]:.10f} hartree")
        
    except Exception as e:
        print(f"LASSIrq-VQE calculation failed: {e}")
        return None
    
    return lassi_vqe

def example_h6_comparison():
    """Example 3: H6 comparison between LASSIrq and standard approach"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: H6 LASSIrq Comparison")
    print("=" * 60)
    
    # Construct LASSIrq instance
    lsi, las = construct_lassirq_instance(h6_sto3g, verbose=1)
    
    # Run LASSIrq calculation
    e_roots, si_rq = lsi.kernel()
    
    print(f"\nH6 LASSIrq[1,1] results:")
    print(f"Number of generated states: {len(e_roots)}")
    print(f"Ground state energy: {e_roots[0]:.10f} hartree")
    print(f"First few excited state energies:")
    for i, e in enumerate(e_roots[1:4]):  # Show first 3 excited states
        print(f"  State {i+1}: {e:.10f} hartree")
    
    print(f"\nGround state composition:")
    print(f"  Dominant coefficients in SI vector: {si_rq[:,0][:5]}")
    
    return lsi, las

def main():
    """Main function to run all examples"""
    print("LASSIrq Implementation Examples")
    print("===============================")
    print("This demonstrates the switch from manual state averaging to automatic")
    print("charge transfer state generation using LASSIrq[r,q] parameters.")
    print("")
    
    results = {}
    
    # Example 1: H4 with LASSIrq
    try:
        results['h4'] = example_h4_lassirq()
        print("\n✓ H4 LASSIrq example completed successfully")
    except Exception as e:
        print(f"\n✗ H4 LASSIrq example failed: {e}")
    
    # Example 2: C2H4N4 with LASSIrq-VQE  
    try:
        results['c2h4n4'] = example_c2h4n4_lassirq_vqe()
        if results['c2h4n4'] is not None:
            print("\n✓ C2H4N4 LASSIrq-VQE example completed successfully")
        else:
            print("\n⚠ C2H4N4 LASSIrq-VQE example completed with issues")
    except Exception as e:
        print(f"\n✗ C2H4N4 LASSIrq-VQE example failed: {e}")
    
    # Example 3: H6 comparison
    try:
        results['h6'] = example_h6_comparison()
        print("\n✓ H6 LASSIrq comparison completed successfully")
    except Exception as e:
        print(f"\n✗ H6 LASSIrq comparison failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF CHANGES")
    print("=" * 60)
    print("✓ Switched from manual LASSI with state_average() to LASSIrq")
    print("✓ Removed all manual state averaging parameters (weights, spins, smults, charges)")
    print("✓ Corrected spinsub values for hydrogen molecules (0 instead of 1 for paired electrons)")
    print("✓ LASSIrq automatically generates charge transfer states with r=1, q=1 parameters")
    print("✓ LASSI_VQE class updated to use LASSIrq for state interaction")
    print("\nKey Benefits:")
    print("- Automatic CT state generation eliminates manual parameter tuning")  
    print("- Physically consistent spin configurations")
    print("- Simplified interface with fewer parameters")
    print("- More robust and systematic approach to LASSI calculations")
    
    return results

if __name__ == "__main__":
    results = main()