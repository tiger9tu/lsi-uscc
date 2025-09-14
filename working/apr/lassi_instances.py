#!/usr/bin/env python3
"""
LASSI Instance Construction for molecules from test_nop_org.py

This script constructs LASSI (Localized Active Space-State Interaction) instances
for the molecular configurations defined in test_nop_org.py, with appropriate
state averaging parameters for each system.
"""

import numpy as np
import copy
from pyscf import gto, scf, lib
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSI, LASSIrq

# Define molecular configurations from test_nop_org.py
# Note: These are defined in the __main__ block of test_nop_org.py, so we redefine them here

# Read geometry files
data_dir = '/home/jinx/repo/qchem/las_uccsd_data'

def read_xyz_file(filepath):
    """Read XYZ file content"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None

# Load external geometries 
stil90xyz = read_xyz_file(f'{data_dir}/stilbene/geometries/stil-90.xyz')
c4xyz = read_xyz_file(f'{data_dir}/polyenes/geometries/c4.xyz') 
c6xyz = read_xyz_file(f'{data_dir}/polyenes/geometries/c6.xyz')
c10xyz = read_xyz_file(f'{data_dir}/polyenes/geometries/c10.xyz')
h10_circle_xyz = read_xyz_file(f'{data_dir}/circle/H10.xyz')

# Internal geometries
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

H8xyz = '''    H      0.000000000000   0.000000000000   0.000000000000
H      1.000000000000   0.000000000000   0.000000000000
H      0.273746762116   2.195450598147   0.100000000000
H      1.232912762116   1.895450598147  -0.100000000000
H      0.507178110854   4.193780995243   0.049334760036
H      1.506140937609   3.988021397347  -0.049334760036
H      0.845946518048   6.364231296231   0.197836732111
H      1.674032054647   5.908472292654  -0.197836732111
'''

# Define molecular configurations
h4_sto3g = {
    'name': 'H4_STO3G',
    'xyz': H4xyz,
    'basis': 'sto-3g',
    'ncas': [2, 2],
    'nelecas': [2, 2],
    'spinsub': [0, 0],  # Corrected spin for H4 (each fragment has paired electrons)
    'frag_atom_list': ((0, 1), (2, 3)),
}

h6_sto3g = {
    'name': 'H6_STO3G',
    'xyz': H6xyz,
    'basis': 'sto-3g',
    'ncas': [2, 2, 2],
    'nelecas': [2, 2, 2],
    'spinsub': [0, 0, 0],  # Corrected spin for H6
    'frag_atom_list': ((0, 1), (2, 3), (4, 5)),
}

h6_631g = copy.deepcopy(h6_sto3g)
h6_631g['name'] = 'H6_631G'
h6_631g['basis'] = '6-31g'

h8_sto3g = {
    'name': 'H8_STO3G',
    'xyz': H8xyz,
    'basis': 'sto-3g',
    'ncas': [2, 2, 2, 2],
    'nelecas': [2, 2, 2, 2],
    'spinsub': [1, 1, 1, 1],
    'frag_atom_list': ((0, 1), (2, 3), (4, 5), (6, 7)),
}

h8_631g = copy.deepcopy(h8_sto3g)
h8_631g['name'] = 'H8_631G'
h8_631g['basis'] = '6-31g'

# External geometry configs (only if files exist)
molecular_configs = [h4_sto3g, h6_sto3g, h6_631g, h8_sto3g, h8_631g]

if stil90xyz:
    stil_sto3g_90 = {
        'name': 'STIL_STO3G_90',
        'xyz': stil90xyz,
        'basis': 'sto-3g',
        'ncas': [4,2,4],
        'nelecas': [4,2,4],
        'spinsub': [1, 1, 1],
        'frag_atom_list': [[1,2,3,4,5,6,15,16,17,18,19], [0,7,14,20], [8,9,10,11,12,13,21,22,23,24,25]],
    }
    molecular_configs.append(stil_sto3g_90)

if c4xyz:
    c4_sto3g = {
        'name': 'C4_STO3G',
        'xyz': c4xyz,
        'basis': 'sto-3g',
        'ncas': [2,2],
        'nelecas': [2,2],
        'spinsub': [1,1],
        'frag_atom_list': [[0,2], [3,1]],
    }
    c4_631g = copy.deepcopy(c4_sto3g)
    c4_631g['name'] = 'C4_631G'
    c4_631g['basis'] = '6-31g'
    molecular_configs.extend([c4_sto3g, c4_631g])

if c6xyz:
    c6_sto3g = {
        'name': 'C6_STO3G',
        'xyz': c6xyz,
        'basis': 'sto-3g',
        'ncas': [2,2,2],
        'nelecas': [2,2,2],
        'spinsub': [1,1,1],
        'frag_atom_list': [[0,2], [10,12], [3,1]],
    }
    c6_631g = copy.deepcopy(c6_sto3g)
    c6_631g['name'] = 'C6_631G'
    c6_631g['basis'] = '6-31g'
    molecular_configs.extend([c6_sto3g, c6_631g])

if c10xyz:
    c10_sto3g = {
        'name': 'C10_STO3G',
        'xyz': c10xyz,
        'basis': 'sto-3g',
        'ncas': [2,2,2,2,2],
        'nelecas': [2,2,2,2,2],
        'spinsub': [1,1,1,1,1],
        'frag_atom_list': [[0,2], [10,12], [18,19], [13,11], [3,1]],
    }
    molecular_configs.append(c10_sto3g)

if h10_circle_xyz:
    h10_circle_sto3g = {
        'name': 'H10_CIRCLE_STO3G',
        'xyz': h10_circle_xyz,
        'basis': 'sto-3g',
        'ncas': [2,2,2,2,2],
        'nelecas': [2,2,2,2,2],
        'spinsub': [1,1,1,1,1],
        'frag_atom_list': [[0,1], [2,3], [4,5], [6,7], [8,9]],
    }
    molecular_configs.append(h10_circle_sto3g)

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

def get_state_averaging_params(mol_config):
    """Define state averaging parameters based on molecular system"""
    name = mol_config['name']
    nfrags = len(mol_config['ncas'])
    
    # For hydrogen chain systems
    if 'H4' in name or 'H6' in name or 'H8' in name or 'H10' in name:
        # Hydrogen systems: ground state + various spin excitations
        if nfrags == 2:  # H4
            weights = [0.7, 0.3]  # Ground state dominant
            spins = [[1, 1], [1, -1]]  # Singlet and triplet-like states
            smults = [[2, 2], [2, 2]]  # All doublets on fragments
            charges = [[0, 0], [0, 0]]  # Neutral fragments
        elif nfrags == 3:  # H6
            weights = [0.5, 0.3, 0.2]  # Ground state + excitations
            spins = [[1, 1, 1], [1, 1, -1], [-1, 1, 1]]
            smults = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
            charges = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        elif nfrags == 4:  # H8
            weights = [0.4, 0.3, 0.2, 0.1]
            spins = [[1, 1, 1, 1], [1, 1, 1, -1], [1, 1, -1, -1], [-1, -1, 1, 1]]
            smults = [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
            charges = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        elif nfrags == 5:  # H10
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]
            spins = [[1, 1, 1, 1, 1], [1, 1, 1, 1, -1], [1, 1, 1, -1, -1], 
                    [1, 1, -1, -1, -1], [-1, -1, -1, 1, 1]]
            smults = [[2, 2, 2, 2, 2]] * 5
            charges = [[0, 0, 0, 0, 0]] * 5
            
    # For carbon systems (polyenes)
    elif 'C4' in name or 'C6' in name or 'C10' in name:
        if nfrags == 2:  # C4
            weights = [0.6, 0.4]  # Ground + first excited
            spins = [[1, 1], [-1, -1]]  # Different spin arrangements
            smults = [[2, 2], [2, 2]]
            charges = [[0, 0], [0, 0]]
        elif nfrags == 3:  # C6
            weights = [0.5, 0.3, 0.2]
            spins = [[1, 1, 1], [1, -1, 1], [-1, 1, -1]]
            smults = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
            charges = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        elif nfrags == 5:  # C10
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]
            spins = [[1, 1, 1, 1, 1], [1, 1, -1, 1, 1], [1, -1, 1, -1, 1],
                    [-1, 1, -1, 1, -1], [1, -1, -1, -1, 1]]
            smults = [[2, 2, 2, 2, 2]] * 5
            charges = [[0, 0, 0, 0, 0]] * 5
            
    # For stilbene system
    elif 'STIL' in name:
        # 3-fragment stilbene: phenyl-vinyl-phenyl
        weights = [0.4, 0.3, 0.2, 0.1]  # Ground + excited states
        spins = [[1, 1, 1], [1, -1, 1], [-1, 1, -1], [1, 1, -1]]
        smults = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        charges = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        
    # Default case - simple ground state
    else:
        nstates = min(4, 2*nfrags)  # Reasonable number of states
        weights = [1.0/nstates] * nstates
        spins = []
        smults = []
        charges = []
        
        for i in range(nstates):
            # Generate different spin configurations
            spin_pattern = [1 if j < nfrags//2 + i%2 else -1 for j in range(nfrags)]
            spins.append(spin_pattern)
            smults.append([2] * nfrags)  # All doublets
            charges.append([0] * nfrags)  # All neutral
    
    return {
        'weights': weights,
        'spins': spins, 
        'smults': smults,
        'charges': charges
    }

def construct_lassi_instance(mol_config, verbose=0):
    """Construct LASSI instance for a given molecular configuration"""
    print(f"\nConstructing LASSI instance for {mol_config['name']}")
    
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
    
    # Get state averaging parameters
    sa_params = get_state_averaging_params(mol_config)
    
    # Setup state averaging
    las_sa = las.state_average(
        sa_params['weights'],
        spins=sa_params['spins'],
        smults=sa_params['smults'], 
        charges=sa_params['charges']
    )
    
    # Run LASCI to get the CI vectors for all states
    las_sa.lasci()
    
    # Create LASSI instance
    lsi = LASSI(las_sa)
    
    if verbose > 0:
        print(f"  LASSCF energy: {las.e_tot:.10f}")
        print(f"  Number of states: {las_sa.nroots}")
        print(f"  State weights: {sa_params['weights']}")
        print(f"  Spins: {sa_params['spins']}")
        print(f"  Smults: {sa_params['smults']}")
        print(f"  Charges: {sa_params['charges']}")
    
    return lsi, las_sa

def main():
    """Main function to construct LASSI instances for all molecules"""
    
    # Use the molecular_configs list defined above
    mol_configs = molecular_configs
    
    lassi_instances = {}
    
    print("=" * 60)
    print("CONSTRUCTING LASSI INSTANCES")
    print("=" * 60)
    
    for mol_config in mol_configs:
        try:
            lsi, las_sa = construct_lassi_instance(mol_config, verbose=1)
            lassi_instances[mol_config['name']] = {
                'lassi': lsi,
                'las_sa': las_sa,
                'mol_config': mol_config
            }
            print(f"✓ Successfully created LASSI instance for {mol_config['name']}")
        except Exception as e:
            print(f"✗ Failed to create LASSI instance for {mol_config['name']}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successfully constructed {len(lassi_instances)} LASSI instances:")
    for name in lassi_instances.keys():
        print(f"  - {name}")
    
    # Example: Run LASSI diagonalization for the first instance
    if lassi_instances:
        example_name = list(lassi_instances.keys())[0]
        example_lsi = lassi_instances[example_name]['lassi']
        
        print(f"\nRunning example LASSI calculation for {example_name}:")
        try:
            e_roots, si = example_lsi.kernel()
            print(f"LASSI energies for {example_name}:")
            for i, e in enumerate(e_roots[:5]):  # Show first 5 energies
                print(f"  State {i}: {e:.10f} hartree")
        except Exception as e:
            print(f"LASSI calculation failed: {e}")
    
    return lassi_instances

if __name__ == "__main__":
    instances = main()