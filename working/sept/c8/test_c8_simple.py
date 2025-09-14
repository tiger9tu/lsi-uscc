#!/usr/bin/env python3
"""
Simple C8 test - just CASCI calculation
"""

import sys
import time
from pyscf import gto, scf, mcscf

# Setup molecule
geom_file = '/home/jinx/repo/qchem/las_uccsd_data/polyenes/geometries/c8.xyz'

print("Setting up C8 molecule...")
with open(geom_file, 'r') as f:
    lines = f.readlines()

# File format is direct atom list, not standard xyz with header
atom_string = ""
for line in lines:
    if line.strip():
        atom_string += line

mol = gto.M(
    atom=atom_string,
    basis='6-31g',
    verbose=1
)
mol.build()

print(f"C8 molecule setup complete:")
print(f"  Atoms: {mol.natm}")
print(f"  Electrons: {mol.nelectron}")
print(f"  Basis functions: {mol.nao}")

# Try mean-field calculation
print("\nRunning RHF calculation...")
start_time = time.time()
mf = scf.RHF(mol)
mf.kernel()
mf_time = time.time() - start_time
print(f"RHF completed in {mf_time:.2f} seconds")
print(f"RHF energy: {mf.e_tot:.10f} hartree")

# Try CASCI calculation
print(f"\nRunning CASCI(8,8) calculation...")
start_time = time.time()
mc = mcscf.CASCI(mf, 8, 8)  # 8 orbitals, 8 electrons
mc.kernel()
casci_time = time.time() - start_time
print(f"CASCI completed in {casci_time:.2f} seconds")
print(f"CASCI energy: {mc.e_tot:.10f} hartree")
print(f"CASCI converged: {mc.converged}")

print(f"\n✅ C8 basic calculation test completed successfully!")