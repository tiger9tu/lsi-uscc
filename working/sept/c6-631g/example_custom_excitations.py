#!/usr/bin/env python3
"""
Example demonstrating how to use LAS-VQE-NOSI with custom excitation parameter sets

This example shows how users can implement requirement 2.a - providing custom sets of excitation parameters.
"""

import numpy as np
from pyscf import gto, lib
from las_vqe_nosi import LASVQENOSI

def example_with_custom_excitation_sets():
    """
    Example showing how to provide custom excitation parameter sets to LAS-VQE-NOSI
    
    This demonstrates requirement 2.a: allow users to implement custom excitation parameters
    """
    
    print("=== LAS-VQE-NOSI with Custom Excitation Parameter Sets ===")
    
    # Define H4 molecule
    h4_xyz = """H      0.000000000000   0.000000000000   0.000000000000
H      1.000000000000   0.000000000000   0.000000000000
H      0.273746762116   2.195450598147   0.100000000000
H      1.232912762116   1.895450598147  -0.100000000000"""
    
    mol = gto.M(atom=h4_xyz, basis='sto-3g', verbose=0)
    
    # Initialize LAS-VQE-NOSI
    calc = LASVQENOSI(
        mol=mol,
        ncas_sub=(2, 2),
        nelec_sub=((1, 1), (1, 1)),
        basis='sto-3g',
        frag_atom_list=((0, 1), (2, 3)),
        gradient_threshold=1e-4,
        vqe_max_cycles=5,
        verbose=1
    )
    
    # Setup molecule and LASSCF
    print("\n1. Setting up molecule and LASSCF...")
    calc.setup_molecule()
    calc.perform_lasscf()
    calc.setup_casci_reference()
    
    # First, let's see what the default gradient-based selection gives us
    print("\n2. Getting default excitation selection for reference...")
    default_sets = calc.get_default_excitation_sets()
    print(f"Default selection produced {len(default_sets)} sets with {[len(s[0]) for s in default_sets]} excitations each")
    
    # Now demonstrate custom excitation parameter sets via direct assignment
    print("\n3. Defining and assigning custom excitation parameter sets...")
    
    # Example custom sets - users can define these based on their chemical intuition
    # These are just examples - real applications would use domain-specific knowledge
    
    # Custom Set 1: First few excitations (e.g., most important based on gradient)
    custom_a_idxs_1 = default_sets[0][0][:10]  # Take first 10 double excitations
    custom_i_idxs_1 = default_sets[0][1][:10]  # Take first 10 single excitations
    
    # Custom Set 2: Different selection (e.g., interfragment excitations)
    custom_a_idxs_2 = default_sets[1][0][:15]  # Take first 15 double excitations from second set
    custom_i_idxs_2 = default_sets[1][1][:15]  # Take first 15 single excitations from second set
    
    # Custom Set 3: Mixed selection (demonstrate flexibility)
    custom_a_idxs_3 = default_sets[0][0][5:20]  # Middle portion of first set
    custom_i_idxs_3 = default_sets[0][1][5:20]  # Middle portion of first set
    
    # Direct assignment to member variable - this is the key change!
    calc.excitation_parameter_sets = [
        (custom_a_idxs_1, custom_i_idxs_1),
        (custom_a_idxs_2, custom_i_idxs_2), 
        (custom_a_idxs_3, custom_i_idxs_3)
    ]
    
    print(f"Assigned 3 custom excitation sets with {[len(s[0]) for s in calc.excitation_parameter_sets]} excitations each")
    
    # Perform VQE optimization on custom sets
    print("\n4. Performing VQE optimization on custom parameter sets...")
    vqe_states = calc.perform_vqe_on_sets()
    
    print(f"VQE optimization completed for {len(vqe_states)} custom states")
    print(f"Individual VQE energies: {[f'{state.energy:.8f}' for state in vqe_states]}")
    
    # Perform state interaction
    print("\n5. Performing state interaction...")
    final_energies, eigenvectors = calc.perform_state_interaction()
    
    print(f"Final ground state energy: {final_energies[0]:.10f} hartree")
    print(f"Excited state energies: {final_energies[1:]}")
    print(f"Ground state mixing coefficients: {eigenvectors[:, 0]}")
    
    return calc.result


def example_fragment_specific_excitations():
    """
    Example showing how to define fragment-specific excitation parameter sets
    
    This demonstrates advanced usage where excitations are selected based on 
    fragment localization or chemical intuition.
    """
    
    print("\n=== Fragment-Specific Excitation Selection ===")
    
    # H6 molecule with 3 fragments
    h6_xyz = """H      0.000000000000   0.000000000000   0.000000000000
H      1.000000000000   0.000000000000   0.000000000000
H      0.273746762116   2.195450598147   0.100000000000
H      1.232912762116   1.895450598147  -0.100000000000
H      0.507178110854   4.193780995243   0.049334760036
H      1.506140937609   3.988021397347  -0.049334760036"""
    
    mol = gto.M(atom=h6_xyz, basis='sto-3g', verbose=0)
    
    calc = LASVQENOSI(
        mol=mol,
        ncas_sub=(2, 2, 2),
        nelec_sub=((1, 1), (1, 1), (1, 1)),
        basis='sto-3g',
        frag_atom_list=((0, 1), (2, 3), (4, 5)),
        gradient_threshold=1e-4,
        vqe_max_cycles=3,  # Shorter for demonstration
        verbose=1
    )
    
    # Setup
    calc.setup_molecule()
    calc.perform_lasscf()
    calc.setup_casci_reference()
    
    # Get default selection for reference
    default_sets = calc.get_default_excitation_sets()
    all_a_idxs = []
    all_i_idxs = []
    for a_set, i_set in default_sets:
        all_a_idxs.extend(a_set)
        all_i_idxs.extend(i_set)
    
    # Define fragment-specific sets (this is where chemical intuition comes in)
    # Fragment orbitals: [0,1] for frag0, [2,3] for frag1, [4,5] for frag2 (spin orbitals would be doubled)
    
    # Set 1: Excitations involving fragments 0-1 (first two fragments)
    frag01_a_idxs = [a for a in all_a_idxs if all(idx < 4 for idx in a)][:8]  # First 8
    frag01_i_idxs = [i for i in all_i_idxs if all(idx < 4 for idx in i)][:8]  # First 8
    
    # Set 2: Excitations involving fragments 1-2 (last two fragments)  
    frag12_a_idxs = [a for a in all_a_idxs if any(idx >= 2 for idx in a)][:10]  # First 10
    frag12_i_idxs = [i for i in all_i_idxs if any(idx >= 2 for idx in i)][:10]  # First 10
    
    # Set 3: Inter-fragment excitations (across all fragments)
    inter_a_idxs = all_a_idxs[:6]  # First 6 overall
    inter_i_idxs = all_i_idxs[:6]  # First 6 overall
    
    # Direct assignment to member variable
    calc.excitation_parameter_sets = [
        (frag01_a_idxs, frag01_i_idxs),
        (frag12_a_idxs, frag12_i_idxs),
        (inter_a_idxs, inter_i_idxs)
    ]
    
    print(f"Fragment-specific sets: {[len(s[0]) for s in calc.excitation_parameter_sets]} excitations each")
    
    # Run with fragment-specific excitation sets
    calc.perform_vqe_on_sets()
    final_energies, _ = calc.perform_state_interaction()
    
    print(f"Final ground state energy with fragment-specific excitations: {final_energies[0]:.10f} hartree")
    
    return calc.result


if __name__ == "__main__":
    # Example 1: Basic custom excitation sets
    result1 = example_with_custom_excitation_sets()
    
    # Example 2: Fragment-specific excitation sets
    result2 = example_fragment_specific_excitations()
    
    print("\n=== Summary ===")
    print("Both examples demonstrate how users can implement requirement 2.a:")
    print("- Direct assignment of custom excitation parameter sets to calc.excitation_parameter_sets")
    print("- Flexibility to define excitation sets based on chemical intuition")
    print("- Fragment-specific or physics-based excitation selection strategies")
    print("- Simple member variable access replaces complex method calls")
    print(f"\nExample 1 final energy: {result1.ground_state_energy:.10f} hartree")
    print(f"Example 2 final energy: {result2.ground_state_energy:.10f} hartree")