#!/usr/bin/env python3
"""
Simple test of LASSI instance construction
"""

import numpy as np
from pyscf import gto, scf, lib
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSI

# Define H4 molecule configuration (same as in test_nop_org.py)
H4xyz = ''' H      0.000000000000   0.000000000000   0.000000000000
H      1.000000000000   0.000000000000   0.000000000000
H      0.273746762116   2.195450598147   0.100000000000
H      1.232912762116   1.895450598147  -0.100000000000
'''

h4_sto3g = {
    'name': 'H4_STO3G',
    'xyz': H4xyz,
    'basis': 'sto-3g',
    'ncas': [2, 2],
    'nelecas': [2, 2],
    'spinsub': [1, 1],
    'frag_atom_list': ((0, 1), (2, 3)),
}

def setup_molecule(mol_config):
    """Setup molecule from configuration"""
    symmetry = mol_config.get('symmetry', None)
    charge = mol_config.get('charge', 0)
    spin = mol_config.get('spin', 0)
    
    mol = gto.M(
        atom=mol_config['xyz'],
        basis=mol_config['basis'],
        symmetry=symmetry,
        charge=charge,
        spin=spin,
        verbose=0,
        output=None
    )
    
    if mol_config.get('HF') == 'ROHF':
        mf = scf.ROHF(mol).run()
    else:
        mf = scf.RHF(mol).run()
    
    return mol, mf

def construct_lassi_instance(mol_config, verbose=0):
    """Construct LASSI instance for H4 with appropriate state averaging"""
    print(f"Constructing LASSI instance for {mol_config['name']}")
    
    # Setup molecule and HF reference
    mol, mf = setup_molecule(mol_config)
    
    # Setup LASSCF with LAS parameters from mol_config
    las = LASSCF(mf, mol_config['ncas'], mol_config['nelecas'], 
                 spin_sub=mol_config['spinsub'])
    
    # Localize initial guess using fragment atom lists
    mo_loc = las.localize_init_guess(mol_config['frag_atom_list'], mf.mo_coeff)
    las.kernel(mo_loc)
    
    if not las.converged:
        print(f"Warning: LASSCF not converged for {mol_config['name']}")
    
    print(f"LASSCF energy: {las.e_tot:.10f}")
    
    # Define state averaging parameters for H4 (2 fragments)
    # Ground state + simple excitation
    weights = [0.7, 0.3]  # Ground state dominant
    spins = [[1, 1], [1, -1]]  # Different spin couplings
    smults = [[2, 2], [2, 2]]  # All doublets on fragments  
    charges = [[0, 0], [0, 0]]  # Neutral fragments
    
    # Setup state averaging
    las_sa = las.state_average(
        weights,
        spins=spins,
        smults=smults, 
        charges=charges
    )
    
    # Run LASCI to get the CI vectors for all states
    las_sa.lasci()
    
    print(f"State-averaged LASSCF completed with {las_sa.nroots} states")
    print(f"State energies: {las_sa.e_states}")
    
    # Create LASSI instance
    lsi = LASSI(las_sa)
    
    if verbose > 0:
        print(f"State weights: {weights}")
        print(f"Spins: {spins}")
        print(f"Smults: {smults}")
        print(f"Charges: {charges}")
    
    return lsi, las_sa

def main():
    """Test LASSI construction and diagonalization"""
    
    print("=" * 50)
    print("TESTING LASSI CONSTRUCTION FOR H4")
    print("=" * 50)
    
    # Construct LASSI instance
    lsi, las_sa = construct_lassi_instance(h4_sto3g, verbose=1)
    
    print("\nLASSI instance created successfully!")
    print(f"Number of fragments: {las_sa.nfrags}")
    print(f"Number of states: {las_sa.nroots}")
    
    # Run LASSI diagonalization
    print("\nRunning LASSI diagonalization...")
    try:
        e_roots, si = lsi.kernel()
        print(f"LASSI diagonalization completed!")
        print(f"Number of LASSI eigenvalues: {len(e_roots)}")
        
        print("\nLASSI eigenvalues:")
        for i, e in enumerate(e_roots):
            print(f"  State {i}: {e:.10f} hartree")
            
        print(f"\nLASSI ground state energy: {e_roots[0]:.10f} hartree")
        print(f"SI vector shape: {si.shape}")
        
        # Show some analysis
        if hasattr(si, 's2'):
            print(f"<S^2> values: {si.s2}")
        if hasattr(si, 'nelec'):
            print(f"Electron numbers: {si.nelec}")
            
    except Exception as e:
        print(f"LASSI diagonalization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)