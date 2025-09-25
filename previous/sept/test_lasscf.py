#!/usr/bin/env python3
"""
Test LASSCF setup for C8
"""

import sys
import time
from pyscf import gto, scf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

def load_geometry(geom_file):
    """Load geometry from file"""
    with open(geom_file, 'r') as f:
        return f.read()

# Setup C8 molecule
geom_file = '/home/jinx/repo/qchem/las_uccsd_data/polyenes/geometries/c8.xyz'
atom_string = load_geometry(geom_file)
mol = gto.M(atom=atom_string, basis='6-31g', verbose=1)
mol.build()

print(f"C8 molecule: {mol.natm} atoms, {mol.nelectron} electrons")

# Mean-field calculation
mf = scf.RHF(mol)
mf.kernel()

# Try simple LASSCF with 2 fragments
print("\nTesting LASSCF with 2 fragments...")
try:
    las = LASSCF(mf, (4, 4), (4, 4), spin_sub=(1, 1))
    print("✅ LASSCF object created successfully")
    
    # State averaging - charges, spins, smults must be nested lists for each state
    las.state_average_(weights=[1.0], charges=[[0, 0]], spins=[[0, 0]], smults=[[1, 1]])
    print("✅ State averaging set successfully")
    
    # Use natural orbitals as initial guess
    las.kernel()
    print(f"✅ LASSCF calculation completed: {las.e_tot:.10f} hartree")
    
except Exception as e:
    print(f"❌ LASSCF failed: {e}")
    import traceback
    traceback.print_exc()