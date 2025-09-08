#!/usr/bin/env python3
"""
LASSIrq Instance Construction for molecules

This script constructs LASSIrq (Localized Active Space-State Interaction with r,q parameters) instances
for various molecular configurations, automatically generating charge transfer states.
"""

import numpy as np
import copy
from pyscf import gto, scf, lib
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSIrq

# Define molecular configurations
# Note: spinsub represents spin multiplicity (2S+1), not spin quantum number S

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

# Define molecular configurations with proper spinsub values (spin multiplicity)
h4_sto3g = {
    'name': 'H4_STO3G',
    'xyz': H4xyz,
    'basis': 'sto-3g',
    'ncas': [2, 2],
    'nelecas': [2, 2],
    'spinsub': [1, 1],  # Spin multiplicity = 2S+1 = 1 (singlet)
    'frag_atom_list': ((0, 1), (2, 3)),
}

h6_sto3g = {
    'name': 'H6_STO3G',
    'xyz': H6xyz,
    'basis': 'sto-3g',
    'ncas': [2, 2, 2],
    'nelecas': [2, 2, 2],
    'spinsub': [1, 1, 1],  # Spin multiplicity = 2S+1 = 1 (singlet)
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
    'spinsub': [1, 1, 1, 1],  # Spin multiplicity = 2S+1 = 1 (singlet)
    'frag_atom_list': ((0, 1), (2, 3), (4, 5), (6, 7)),
}

h8_631g = copy.deepcopy(h8_sto3g)
h8_631g['name'] = 'H8_631G'
h8_631g['basis'] = '6-31g'

# Base molecular configs
molecular_configs = [h4_sto3g, h6_sto3g, h6_631g, h8_sto3g, h8_631g]

# External geometry configs (only if files exist)
if stil90xyz:
    stil_sto3g_90 = {
        'name': 'STIL_STO3G_90',
        'xyz': stil90xyz,
        'basis': 'sto-3g',
        'ncas': [4,2,4],
        'nelecas': [4,2,4],
        'spinsub': [1, 1, 1],  # Spin multiplicity = 2S+1 = 1 (singlet)
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
        'spinsub': [1,1],  # Spin multiplicity = 2S+1 = 1 (singlet)
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
        'spinsub': [1,1,1],  # Spin multiplicity = 2S+1 = 1 (singlet)
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
        'spinsub': [1,1,1,1,1],  # Spin multiplicity = 2S+1 = 1 (singlet)
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
        'spinsub': [1,1,1,1,1],  # Spin multiplicity = 2S+1 = 1 (singlet)
        'frag_atom_list': [[0,1], [2,3], [4,5], [6,7], [8,9]],
    }
    molecular_configs.append(h10_circle_sto3g)

def setup_molecule(mol_config):
    """Setup molecule from configuration"""
    # symmetry = mol_config.get('symmetry', None)
    # charge = mol_config.get('charge', 0)
    # spin = mol_config.get('spin', 0)
    
    mol = gto.M(
        atom=mol_config['xyz'],
        basis=mol_config['basis'],
        # symmetry=symmetry,
        # charge=charge,
        # spin=spin,
        verbose=0,
        output=None
    )
    
    if mol_config.get('HF') == 'ROHF':
        mf = scf.ROHF(mol).run()
    else:
        mf = scf.RHF(mol).run()
    
    return mol, mf

def get_lassirq_params(mol_config):
    """Define LASSIrq parameters based on molecular system"""
    name = mol_config['name']
    nfrags = len(mol_config['ncas'])
    
    # Use r=1, q=1 for most systems (1-electron charge transfer)
    # Can be customized based on molecular system if needed
    r = 1
    q = 1
    
    # For larger systems, might want to use r=2 for 2-electron transfers
    if nfrags >= 4:
        # For systems with 4+ fragments, can also consider 2-electron transfers
        # r = 2  # Uncomment if 2-electron transfers are desired
        pass
    
    return {
        'r': r,
        'q': q
    }

def construct_lassirq_instance(mol_config, verbose=0):
    """Construct LASSIrq instance for a given molecular configuration"""
    print(f"\nConstructing LASSIrq instance for {mol_config['name']}")
    
    # Setup molecule and HF reference
    mol, mf = setup_molecule(mol_config)
    
    # Setup LASSCF with LAS parameters from mol_config (no state averaging)
    las = LASSCF(mf, mol_config['ncas'], mol_config['nelecas'], 
                 spin_sub=mol_config['spinsub'])
    
    # Localize initial guess using fragment atom lists
    mo_loc = las.localize_init_guess(mol_config['frag_atom_list'], mf.mo_coeff)
    las.kernel(mo_loc)
    
    if not las.converged:
        print(f"Warning: LASSCF not converged for {mol_config['name']}")
    
    # Get LASSIrq parameters
    lrq_params = get_lassirq_params(mol_config)
    
    # Create LASSIrq instance for automatic CT state generation
    lsi = LASSIrq(las, r=lrq_params['r'], q=lrq_params['q'])
    e_roots, si_rq = lsi.kernel()
    
    if verbose > 0:
        print(f"  LASSCF energy: {las.e_tot:.10f}")
        print(f"  LASSIrq[{lrq_params['r']},{lrq_params['q']}] parameters")
        print(f"  Number of states generated: {len(e_roots)}")
        print(f"  Ground state energy: {e_roots[0]:.10f}")
    
    return lsi, las

def main():
    """Main function to construct LASSIrq instances for all molecules"""
    
    # Use the molecular_configs list defined above
    mol_configs = molecular_configs
    
    lassirq_instances = {}
    
    print("=" * 60)
    print("CONSTRUCTING LASSIrq INSTANCES")
    print("=" * 60)
    
    for mol_config in mol_configs:
        try:
            lsi, las = construct_lassirq_instance(mol_config, verbose=1)
            lassirq_instances[mol_config['name']] = {
                'lassirq': lsi,
                'las': las,
                'mol_config': mol_config
            }
            print(f"✓ Successfully created LASSIrq instance for {mol_config['name']}")
        except Exception as e:
            print(f"✗ Failed to create LASSIrq instance for {mol_config['name']}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successfully constructed {len(lassirq_instances)} LASSIrq instances:")
    for name in lassirq_instances.keys():
        print(f"  - {name}")
    
    # Example: Run LASSIrq diagonalization for the first instance
    if lassirq_instances:
        example_name = list(lassirq_instances.keys())[0]
        example_lsi = lassirq_instances[example_name]['lassirq']
        
        print(f"\nRunning example LASSIrq calculation for {example_name}:")
        try:
            e_roots, si = example_lsi.kernel()
            print(f"LASSIrq energies for {example_name}:")
            for i, e in enumerate(e_roots[:5]):  # Show first 5 energies
                print(f"  State {i}: {e:.10f} hartree")
        except Exception as e:
            print(f"LASSIrq calculation failed: {e}")
    
    return lassirq_instances

if __name__ == "__main__":
    instances = main()