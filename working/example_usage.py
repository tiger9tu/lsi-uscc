#!/usr/bin/env python3
"""
Example usage of the LASSI_VQE class

This script demonstrates how to use the LASSI_VQE class to perform
Localized Active Space State Interaction with Variational Quantum Eigensolver
optimization on the C2H4N4 molecule.
"""

from lassi_vqe_class import LASSI_VQE
from c2h4n4_struct import structure as struct


def main():
    """Run LASSI-VQE calculation on C2H4N4"""
    
    # Define molecule structure
    mol_func = lambda charge, spin, basis: struct(charge, spin, basis)
    
    # Initialize LASSI-VQE calculation
    lassi_vqe = LASSI_VQE(
        mol=mol_func,
        ncas_sub=(3, 3),
        nelec_sub=((2, 1), (1, 2)),
        basis='6-31g',
        output_file='c2h4n4_lassi_vqe.log'
    )
    
    # Run the complete calculation
    try:
        final_energies, final_eigenvectors, lassi_energies = lassi_vqe.run_full_calculation()
        
        # Print summary results
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        print(f"VQE-LASSI Ground State Energy: {final_energies[0]:.8f} hartree")
        print(f"Standard LASSI Ground State Energy: {lassi_energies[0]:.8f} hartree")
        print(f"Energy Difference: {final_energies[0] - lassi_energies[0]:.8f} hartree")
        
        print(f"\nAll VQE-LASSI Energies: {final_energies}")
        print(f"All Standard LASSI Energies: {lassi_energies}")
        
        # Save molecular orbital files
        lassi_vqe.save_molden_files('c2h4n4_lassi_vqe')
        
        return final_energies, lassi_energies
        
    except Exception as e:
        print(f"Calculation failed with error: {e}")
        raise


def run_step_by_step_example():
    """Example showing step-by-step usage of the class"""
    
    print("\n" + "="*60)
    print("STEP-BY-STEP EXAMPLE")
    print("="*60)
    
    mol_func = lambda charge, spin, basis: struct(charge, spin, basis)
    
    # Initialize
    lassi_vqe = LASSI_VQE(
        mol=mol_func,
        ncas_sub=(3, 3),
        nelec_sub=((2, 1), (1, 2)),
        basis='6-31g',
        output_file='c2h4n4_step_by_step.log'
    )
    
    # Step 1: Setup molecule
    lassi_vqe.setup_molecule()
    print(f"RHF Energy: {lassi_vqe.mf.e_tot:.8f}")
    
    # Step 2: Setup reference LASSCF
    lassi_vqe.setup_lasscf_reference()
    print(f"LASSCF Energy: {lassi_vqe.las.e_tot:.8f}")
    
    # Step 3: Setup CASCI for comparison
    lassi_vqe.setup_casci_reference()
    print(f"CASCI Energy: {lassi_vqe.mc_casci.e_tot:.8f}")
    
    # Step 4: Setup charge transfer states
    e_states = lassi_vqe.setup_charge_transfer_states()
    print(f"Number of LAS states: {len(e_states)}")
    
    # Step 5: Compute effective Hamiltonian
    ham_eff, ovlp_eff = lassi_vqe.compute_effective_hamiltonian()
    print(f"Effective Hamiltonian shape: {ham_eff.shape}")
    
    # Step 6: Build VQE wavefunctions
    psis = lassi_vqe.build_vqe_wavefunctions()
    print(f"Built {len(psis)} VQE wavefunctions")
    
    # Step 7: Compute matrix elements
    H_matrix, S_matrix = lassi_vqe.compute_vqe_matrix_elements()
    print(f"H matrix shape: {H_matrix.shape}")
    print(f"S matrix shape: {S_matrix.shape}")
    
    # Step 8: Solve eigenvalue problem
    energies, eigenvectors = lassi_vqe.solve_generalized_eigenvalue_problem()
    print(f"Ground state energy: {energies[0]:.8f}")
    
    # Step 9: Compare with LASSI
    lassi_energies, _ = lassi_vqe.compare_with_lassi()
    print(f"LASSI ground state energy: {lassi_energies[0]:.8f}")
    
    return energies, lassi_energies


if __name__ == "__main__":
    # Run full calculation
    print("Running full LASSI-VQE calculation...")
    final_energies, lassi_energies = main()
    
    # Optionally run step-by-step example
    run_step_by_step = False  # Set to True to run step-by-step example
    if run_step_by_step:
        print("\n\nRunning step-by-step example...")
        step_energies, step_lassi = run_step_by_step_example()