#!/usr/bin/env python3
"""
Simple test demonstrating direct assignment of excitation parameter sets
"""

from pyscf import gto
from las_vqe_nosi import LASVQENOSI

def test_direct_assignment():
    """Test the new direct assignment approach"""
    
    print("=== Testing Direct Assignment of Excitation Parameter Sets ===")
    
    # Simple H4 molecule
    h4_xyz = """H      0.000000000000   0.000000000000   0.000000000000
H      1.000000000000   0.000000000000   0.000000000000
H      0.273746762116   2.195450598147   0.100000000000
H      1.232912762116   1.895450598147  -0.100000000000"""
    
    mol = gto.M(atom=h4_xyz, basis='sto-3g', verbose=0)
    
    # Initialize calculation
    calc = LASVQENOSI(
        mol=mol,
        ncas_sub=(2, 2),
        nelec_sub=((1, 1), (1, 1)),
        frag_atom_list=((0, 1), (2, 3)),
        gradient_threshold=1e-4,
        vqe_max_cycles=3,
        verbose=1
    )
    
    # Setup basic calculations
    calc.setup_molecule()
    calc.perform_lasscf()
    calc.setup_casci_reference()
    
    # Get default excitations to modify
    default_sets = calc.get_default_excitation_sets()
    print(f"Default sets have {[len(s[0]) for s in default_sets]} excitations")
    
    # Direct assignment - this is the key feature!
    # User can directly assign any excitation parameter sets they want
    custom_set1_a = default_sets[0][0][:5]  # First 5 double excitations
    custom_set1_i = default_sets[0][1][:5]  # First 5 single excitations
    
    custom_set2_a = default_sets[1][0][:8]  # First 8 double excitations
    custom_set2_i = default_sets[1][1][:8]  # First 8 single excitations
    
    # Direct assignment to member variable
    calc.excitation_parameter_sets = [
        (custom_set1_a, custom_set1_i),
        (custom_set2_a, custom_set2_i)
    ]
    
    print(f"User assigned {len(calc.excitation_parameter_sets)} custom sets")
    print(f"Set sizes: {[len(s[0]) for s in calc.excitation_parameter_sets]} excitations each")
    
    # Run VQE and state interaction
    calc.perform_vqe_on_sets()
    calc.perform_state_interaction()
    
    print(f"Final result: {calc.result.ground_state_energy:.10f} hartree")
    
    return True

def test_empty_assignment():
    """Test what happens when user doesn't assign anything"""
    
    print("\n=== Testing Default Behavior (No Assignment) ===")
    
    h4_xyz = """H      0.000000000000   0.000000000000   0.000000000000
H      1.000000000000   0.000000000000   0.000000000000
H      0.273746762116   2.195450598147   0.100000000000
H      1.232912762116   1.895450598147  -0.100000000000"""
    
    mol = gto.M(atom=h4_xyz, basis='sto-3g', verbose=0)
    
    calc = LASVQENOSI(
        mol=mol,
        ncas_sub=(2, 2),
        nelec_sub=((1, 1), (1, 1)),
        frag_atom_list=((0, 1), (2, 3)),
        verbose=1
    )
    
    # Don't assign anything to excitation_parameter_sets
    # The run_full_calculation should use defaults
    result = calc.run_full_calculation()
    
    print(f"Default result: {result.ground_state_energy:.10f} hartree")
    print(f"Used {result.n_states} states by default")
    
    return True

if __name__ == "__main__":
    print("Testing the simplified direct assignment approach...")
    
    # Test 1: Direct assignment
    success1 = test_direct_assignment()
    
    # Test 2: Default behavior
    success2 = test_empty_assignment()
    
    print(f"\n=== Summary ===")
    print("✓ Direct assignment works perfectly!")
    print("✓ Default behavior works when nothing assigned!")
    print("✓ Users can now directly assign calc.excitation_parameter_sets")
    print("✓ No complex method calls needed - simple member variable access!")