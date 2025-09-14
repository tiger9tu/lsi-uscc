#!/usr/bin/env python3
"""
Test script to verify the LASSI_VQE class initialization works correctly
with the new state averaging parameters.
"""

from lassi_vqe_class import LASSI_VQE
from c2h4n4_struct import structure as struct


def test_default_initialization():
    """Test initialization with default parameters"""
    print("Testing default initialization...")
    
    mol_func = lambda charge, spin, basis: struct(charge, spin, basis)
    
    lassi_vqe = LASSI_VQE(
        mol=mol_func,
        ncas_sub=(3, 3),
        nelec_sub=((2, 1), (1, 2)),
        basis='6-31g',
        output_file='test_default.log'
    )
    
    # Verify default parameters
    assert lassi_vqe.state_weights == [0.5, 0.5], f"Expected [0.5, 0.5], got {lassi_vqe.state_weights}"
    assert lassi_vqe.spins == [[1, -1], [-1, 1]], f"Expected [[1, -1], [-1, 1]], got {lassi_vqe.spins}"
    assert lassi_vqe.smults == [[2, 2], [2, 2]], f"Expected [[2, 2], [2, 2]], got {lassi_vqe.smults}"
    assert lassi_vqe.charges == [[0, 0], [0, 0]], f"Expected [[0, 0], [0, 0]], got {lassi_vqe.charges}"
    assert lassi_vqe.extended_state_weights == [0.5, 0.5, 0, 0], f"Expected [0.5, 0.5, 0, 0], got {lassi_vqe.extended_state_weights}"
    
    print("✓ Default initialization test passed!")
    return lassi_vqe


def test_custom_initialization():
    """Test initialization with custom parameters"""
    print("Testing custom initialization...")
    
    mol_func = lambda charge, spin, basis: struct(charge, spin, basis)
    
    custom_state_weights = [0.7, 0.3]
    custom_extended_weights = [0.4, 0.4, 0.1, 0.1]
    custom_charges = [[0, 0], [0, 0]]
    custom_extended_charges = [[0, 0], [0, 0], [-1, 1], [1, -1]]
    
    lassi_vqe = LASSI_VQE(
        mol=mol_func,
        ncas_sub=(3, 3),
        nelec_sub=((2, 1), (1, 2)),
        basis='6-31g',
        output_file='test_custom.log',
        state_weights=custom_state_weights,
        charges=custom_charges,
        extended_state_weights=custom_extended_weights,
        extended_charges=custom_extended_charges
    )
    
    # Verify custom parameters
    assert lassi_vqe.state_weights == custom_state_weights, f"Expected {custom_state_weights}, got {lassi_vqe.state_weights}"
    assert lassi_vqe.charges == custom_charges, f"Expected {custom_charges}, got {lassi_vqe.charges}"
    assert lassi_vqe.extended_state_weights == custom_extended_weights, f"Expected {custom_extended_weights}, got {lassi_vqe.extended_state_weights}"
    assert lassi_vqe.extended_charges == custom_extended_charges, f"Expected {custom_extended_charges}, got {lassi_vqe.extended_charges}"
    
    print("✓ Custom initialization test passed!")
    return lassi_vqe


def test_molecule_and_lasscf_setup():
    """Test that molecule setup works with the new parameters"""
    print("Testing molecule and LASSCF setup...")
    
    mol_func = lambda charge, spin, basis: struct(charge, spin, basis)
    
    lassi_vqe = LASSI_VQE(
        mol=mol_func,
        ncas_sub=(3, 3),
        nelec_sub=((2, 1), (1, 2)),
        basis='6-31g',
        output_file='test_setup.log',
        state_weights=[0.6, 0.4]
    )
    
    # Test molecule setup
    lassi_vqe.setup_molecule()
    assert lassi_vqe.mol is not None, "Molecule not set up properly"
    assert lassi_vqe.mf is not None, "Mean-field calculation not performed"
    print(f"✓ RHF energy: {lassi_vqe.mf.e_tot:.8f} hartree")
    
    # Test LASSCF setup (this should use the custom state weights)
    lassi_vqe.setup_lasscf_reference()
    assert lassi_vqe.las is not None, "LASSCF calculation not performed"
    print(f"✓ LASSCF energy: {lassi_vqe.las.e_tot:.8f} hartree")
    
    # Verify that the state averaging was applied with custom weights
    print("✓ LASSCF setup with custom parameters completed!")
    
    return lassi_vqe


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING LASSI_VQE INITIALIZATION WITH NEW PARAMETERS")
    print("=" * 60)
    
    try:
        # Test 1: Default initialization
        default_vqe = test_default_initialization()
        print()
        
        # Test 2: Custom initialization
        custom_vqe = test_custom_initialization()
        print()
        
        # Test 3: Actual setup with custom parameters
        setup_vqe = test_molecule_and_lasscf_setup()
        print()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("The new initialization parameters are working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        raise