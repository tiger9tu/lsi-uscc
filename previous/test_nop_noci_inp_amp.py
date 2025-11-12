# test the distribution of the nearest neighbor excitations

import numpy as np
import pyscf
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad, lasci_ominus1
from mrh.exploratory.citools import fockspace


from typing import Iterable, Union
from typing import Dict, List, Any, TypedDict
import copy
import pickle
from enum import Enum
from scipy.linalg import eigh
from scipy import optimize
import sys

VERBOSE = 0
if len(sys.argv) > 1:
    try:
        AMPLITUDE = float(sys.argv[1])
        print(f"Using AMPLITUDE argument: {AMPLITUDE}")
    except ValueError:
        print("Invalid AMPLITUDE argument, using default value 1.")
        AMPLITUDE = 1
else:
    AMPLITUDE = 1

def flatten(seq: Iterable) -> list:
    result = []
    for item in seq:
        if isinstance(item, (list, tuple, np.ndarray)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def cilas2f(lasci, norb_f, nelec_f):
    ''' nelec (na, nb)'''
    ci_f = []
    for i, ci in enumerate(lasci):
        ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
    return ci_f

class MolConfig(TypedDict):
    # basic
    name : str
    xyz: str
    basis: str
    ncas: List[int]
    nelecas: List[int]
    spinsub: List[int]
    frag_atom_list: Any  # Can be List or Tuple depending on usage
    frag_spin_orb: Any
    adj: List[List[int]]

class TestConfig(TypedDict):
    epsilon: float
    # nearest neighbor parameters
    nn: bool
    # non-orthogonal configuration interaction parameters
    init_method: str # 'random' or 'uscc_opt' or 'load x'
    max_nx: int  # max number of excitations per configuration
    min_nc: int
    # test choices
    grad_test: bool
    noci_test: bool  
    frozen: str

def get_Sij(psi_i, psi_j, h):
    """Compute the overlap Sij = <psi_i|psi_j>."""
    ucj, _ = psi_j.hc_x(psi_j.x, h)[1:3]
    uci, _ = psi_i.hc_x(psi_i.x, h)[1:3]
    ucj, uci = ucj.ravel(), uci.ravel()
    Sij = uci.conj().dot(ucj)
    return Sij


def get_Sij_Hij(psi_i, psi_j, h):
    ucj, hucj = psi_j.hc_x (psi_j.x, h)[1:3]
    uci, huci = psi_i.hc_x (psi_i.x, h)[1:3]
    ucj, hucj = ucj.ravel (), hucj.ravel ()
    uci = uci.ravel ()
    Sij = uci.conj ().dot (ucj)
    Hij = uci.conj ().dot (hucj)
    return Sij, Hij


def arr_split(arr, n):
    """Split an array into n nearly equal parts."""
    k, m = divmod(len(arr), n)
    return list(arr[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def print_matrix(mat):
    for row in mat:
        print("  ".join(f"{x:.17f}" for x in row))

def zip_excitations(a_idxs, i_idxs):
    excitations = []
    for a, i in zip(a_idxs, i_idxs):
        excitations.append((tuple(i), tuple(a[::-1])))
    return excitations

def print_excitations(a_idxs, i_idxs):
    excitations = zip_excitations(a_idxs, i_idxs)
    for excitation in excitations:
        print([[int(x) for x in flatten(excitation)]])


def nci_test(a_idxs_selected, i_idxs_selected, test_config, mol_config, mol,las, mc_uscc, mf):

    nc = len(a_idxs_selected) + 1
    # we create ci for each excitation
    if VERBOSE >= 1:
        print("Number of CIs: ", nc)

    h1eff,e_core= mc_uscc.get_h1eff(mc_uscc.mo_coeff)
    h2eff = mc_uscc.get_h2eff()

    las_ucc_trial_cis = []
    
    min_ci_eng = 99999.9
    for i in range(nc):
        if( i == nc -1):
            amplitude = 0 # the last one is the reference state
        else:
            amplitude = AMPLITUDE # we should try different amplitudes

        mc_uscc_ci = mcscf.CASCI(mf, sum(mol_config['ncas']), sum(mol_config['nelecas']))
        mc_uscc_ci.mo_coeff = las.mo_coeff
        mc_uscc_ci.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
        mc_uscc_ci.fcisolver.norb_f = mol_config['ncas']
        fci = mc_uscc_ci.fcisolver
        las_ci0_f = cilas2f(las.ci, mol_config['ncas'], mol_config['nelecas'])

        norb = sum(mol_config['ncas']) # not sure about this
        nelec = sum(mol_config['nelecas']) # not sure about this
        norb_f = getattr (fci, 'norb_f', [norb])
        psi =  getattr (fci, 'psi', fci.build_psi (las_ci0_f, norb, norb_f, nelec, frozen=test_config['frozen']))
        
        if i < nc - 1:
            psi.x[psi.nconstr + i] = amplitude

        energy = psi.energy_tot(psi.x, [e_core, h1eff, h2eff])
        if energy < min_ci_eng:
            min_ci_eng = energy

        las_ucc_trial_cis.append(psi)
        
    
    S = np.zeros((nc, nc), dtype=np.complex128)
    H = np.zeros((nc, nc), dtype=np.complex128)
    h = [e_core, h1eff, h2eff]


    # Build S matrix incrementally, discarding linearly dependent CIs
    selected_indices = []
    S_inc = np.zeros((0, 0), dtype=np.complex128)
    threshold = 1e5  # You can adjust this threshold as needed

    for idx, psi in enumerate(las_ucc_trial_cis):
        # Build S matrix for current selection + this CI
        n_sel = len(selected_indices)
        S_new = np.zeros((n_sel + 1, n_sel + 1), dtype=np.complex128)
        # Fill previous block
        if n_sel > 0:
            S_new[:n_sel, :n_sel] = S_inc
            for j, jidx in enumerate(selected_indices):
                S_new[n_sel, j] = get_Sij(psi, las_ucc_trial_cis[jidx], h)
                S_new[j, n_sel] = get_Sij(las_ucc_trial_cis[jidx], psi, h)
        # Diagonal element
        S_new[n_sel, n_sel] = get_Sij(psi, psi, h)
        # Check condition number
        cond = np.linalg.cond(S_new)
        if cond < threshold:
            selected_indices.append(idx)
            S_inc = S_new
        else:
            if VERBOSE >= 1:
                print(f"Discarding CI {idx} due to linear dependence (cond={cond:.2e})")

    # Now, only keep selected CIs for S and H
    nc_sel = len(selected_indices)
    S = np.zeros((nc_sel, nc_sel), dtype=np.complex128)
    H = np.zeros((nc_sel, nc_sel), dtype=np.complex128)
    for i, idx_i in enumerate(selected_indices):
        for j, idx_j in enumerate(selected_indices):
            S[i, j], H[i, j] = get_Sij_Hij(las_ucc_trial_cis[idx_i], las_ucc_trial_cis[idx_j], h)

    if VERBOSE >= 0:
        print(f"Selected {nc_sel} CIs out of {nc} total CIs")
        print("Condition number of S matrix: ", np.linalg.cond(S))

    if VERBOSE >= 1:
        print("S matrix:")
        print_matrix(S)
        print("H matrix:")
        print_matrix(H)

    # print("min_ci_eng: ", min_ci_eng)
    eigvals, eigvecs = eigh(H, S)
    return eigvals[0], eigvecs[0], min_ci_eng   # Take the lowest eigenvalue as the energy



def test(mol_config, test_config,las, mol, mf):
    result = {}
    all_g, g_sel, a_idxs_selected_all, i_idxs_selected_all = grad.get_grad_exact(las, test_config['epsilon'])

    if test_config['grad_test']:
        result['tot_g'] = all_g

    a_idxs_selected_nn = []
    i_idxs_selected_nn = []
    
    nn_g = []
    if(test_config['nn']):
        nn_g, a_idxs_selected_nn, i_idxs_selected_nn = get_nn_excitations(a_idxs_selected_all, i_idxs_selected_all, all_g, mol_config)
        if test_config['grad_test']:
            result['nn_g'] = nn_g

    result['tot_excitation_count'] = len(a_idxs_selected_all)
    result['excitation_count_nn'] = len(a_idxs_selected_nn)

    #Computing energy through the LAS-UCC kernel using selected excitations
    #==========================================================================================
    
    a_idxs_selected = None
    i_idxs_selected = None

    if test_config['nn']:
        a_idxs_selected = a_idxs_selected_nn
        i_idxs_selected = i_idxs_selected_nn
    else:
        a_idxs_selected = a_idxs_selected_all
        i_idxs_selected = i_idxs_selected_all
    
    if VERBOSE >= 2:
        print("All selected excitations:")
        print_excitations(a_idxs_selected, i_idxs_selected)

    mc_uscc = mcscf.CASCI(mf, sum(mol_config['ncas']), sum(mol_config['nelecas']))
    mc_uscc.mo_coeff = las.mo_coeff
    mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
    mc_uscc.fcisolver.norb_f = mol_config['ncas'] # number of orbitals in each fragment
    # easily hit the maximal memory limit
    mc_uscc.fcisolver.frozen = test_config['frozen'] if 'frozen' in test_config else None  
    mc_uscc.kernel(ci0 = cilas2f(las.ci, mol_config['ncas'], mol_config['nelecas']))
    if not mc_uscc.converged:
        print('Warning: kernel hasn\'t converged')
    
    if VERBOSE >= 1:
        print("MC-USCC amplitudes: ", mc_uscc.fcisolver.psi.x)

    result['las_uscc_eng'] =  mc_uscc.e_tot 
   
    if test_config['noci_test']:
        result['las_uscc_noci_eng'], result['las_uscc_noci_vec'], result['min_ci_eng'] = nci_test(a_idxs_selected, i_idxs_selected, test_config, mol_config, mol,las, mc_uscc, mf)
    
    return result
   



def batch_test(mol_config, test_configs):
    results = []

    # Initializing the molecule with RHF
    #===================================
    symmetry = None
    charge = None
    spin = None
    if 'symmetry' in mol_config:
        symmetry = mol_config['symmetry']
    if 'charge' in mol_config:
        charge = mol_config['charge']
    if 'spin' in mol_config:
        spin = mol_config['spin']

    mol = gto.M(atom=mol_config['xyz'], basis=mol_config['basis'], symmetry = symmetry,
        charge = charge, spin = spin,verbose=0,output=None)
    if 'HF' in mol_config and mol_config['HF'] == 'ROHF':
        mf = scf.ROHF(mol).run()
    else:
        mf = scf.RHF(mol).run()

    # Running LASSCF
    #===================================
    las = LASSCF(mf, mol_config['ncas'], mol_config['nelecas'], spin_sub=mol_config['spinsub'], ouput=None)
    mo_loc = las.localize_init_guess(mol_config['frag_atom_list'], mf.mo_coeff)
    las.kernel(mo_loc)
    
    ref = mcscf.CASSCF(mf, sum(mol_config['ncas']), sum(mol_config['nelecas'])).run() # = FCI

    # mc_uscc = mcscf.CASCI(mf, sum(mol_config['ncas']), sum(mol_config['nelecas']))
    # mc_uscc.mo_coeff = las.mo_coeff
    for test_config in test_configs:
        
        result = test(mol_config, test_config, las, mol, mf)
        results.append(result)
    
    return results, ref.e_tot, las.e_tot



def empty_adj(n):
    return [[0 for _ in range(n)] for _ in range(n)]

def circle_adj(n):
    return [[1 if abs(i - j) == 1 or abs(i - j) == n-1 else 0 for j in range(n)] for i in range(n)]


if __name__ == "__main__":
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

    data_dir = '../working/geom'


    
    with open(data_dir + '/stil-001.xyz', 'r', encoding='utf-8') as f:
        stil001xyz = f.read()

    with open(data_dir + '/stil-60.xyz', 'r', encoding='utf-8') as f:
        stil60xyz = f.read()

    with open(data_dir + '/stil-90.xyz', 'r', encoding='utf-8') as f:
        stil90xyz = f.read()

    with open(data_dir + '/stil-120.xyz', 'r', encoding='utf-8') as f:
        stil120xyz = f.read()

    with open(data_dir + '/stil-180.xyz', 'r', encoding='utf-8') as f:
        stil180xyz = f.read()       

    with open(data_dir + '/c4.xyz', 'r', encoding='utf-8') as f:
        c4xyz = f.read()

    with open(data_dir + '/c6.xyz', 'r', encoding='utf-8') as f:
        c6xyz = f.read()

    with open(data_dir + '/c10.xyz', 'r', encoding='utf-8') as f:
        c10xyz = f.read()

    with open(data_dir + '/h10.xyz', 'r', encoding='utf-8') as f:
        h10_circle_xyz = f.read()

    # with open(data_dir + '/kremer/kremer-geometry.xyz', 'r', encoding='utf-8') as f:
    #     kremer_xyz = f.read()

    # kremer_basis = {
    #     "Cr": "def2-tzvp",
    #     "O": "def2-svp",
    #     "N": "def2-svp",
    #     "C": "def2-svp",
    #     "H": "def2-svp",
    # }

    # kremer_def2 : MolConfig = {
    #     'name' : 'kremer_def2',
    #     'HF' : 'ROHF',
    #     'xyz' : kremer_xyz,
    #     'basis' : kremer_basis,
    #     'symmetry' : False,
    #     'charge' : 3,
    #     'spin' : 6,
    #     'ncas': [3,3],
    #     'nelecas': [[3,0],[0,3]],
    #     'spinsub': [4,4],
    #     'frag_atom_list': [[0],[1]]
    # }

    h4_sto3g : MolConfig = {
        'name': 'H4_STO3G',
        'xyz': H4xyz,
        'basis': 'sto-3g',
        'ncas': [2, 2],
        'nelecas': [2, 2],
        'spinsub': [1, 1],
        'frag_atom_list': ((0, 1), (2, 3)),
    }

    
    h6_sto3g : MolConfig = {
        'name': 'H6_STO3G',
        'xyz': H6xyz,
        'basis': 'sto-3g',
        'ncas': [2, 2, 2],
        'nelecas': [2, 2, 2],
        'spinsub': [1, 1, 1],
        'frag_atom_list': ((0, 1), (2, 3), (4, 5)),
        'frag_spin_orb': {
            0: (0, 1, 6, 7),
            1: (2, 3, 8, 9),
            2: (4, 5, 10, 11)
        },
        'adj': empty_adj(3),
    }

    h6_631g = copy.deepcopy(h6_sto3g)
    h6_631g['name'] = 'H6_631G'
    h6_631g['basis'] = '6-31g'

    h6_sto3g_a4 : MolConfig = {
        'name': 'H6_STO3G_A4',
        'xyz': H6xyz,
        'basis': 'sto-3g',
        'ncas': [2, 2],
        'nelecas': [2, 2],
        'spinsub': [1, 1],
        'frag_atom_list': ((0, 1), (2, 3)),
        'frag_spin_orb': {
            0: (0, 1, 6, 7),
            1: (2, 3, 8, 9),
            2: (4, 5, 10, 11)
        },
        'adj': empty_adj(3),
    }
    
    h8_sto3g : MolConfig = {
        'name': 'H8_STO3G',
        'xyz': H8xyz,
        'basis': 'sto-3g',
        'ncas': [2, 2, 2, 2],
        'nelecas': [2, 2, 2, 2],
        'spinsub': [1, 1, 1,1],
        'frag_atom_list': ((0, 1), (2, 3), (4, 5), (6, 7)),
        'frag_spin_orb': {
            0: (0, 1, 8,9),
            1: (2, 3, 10,11),
            2: (4, 5, 12, 13),
            3: (6, 7, 14, 15)
        },
        'adj': empty_adj(4),
    }

    h8_631g = copy.deepcopy(h8_sto3g)
    h8_631g['name'] = 'H8_631G'
    h8_631g['basis'] = '6-31g'



    stil_631g_001 : MolConfig = {
        'name': 'STIL_631G_001',
        'xyz': stil001xyz,
        'basis': '631g',
        'ncas': [4,2,4],
        'nelecas': [4,2,4],
        'spinsub': [1, 1, 1],
        'frag_atom_list': [ [1,2,3,4,5,6,15,16,17,18,19] , [0,7, 14,20] , [8,9,10,11,12,13, 21,22,23,24,25] ],
        'frag_spin_orb': {
            0: (0, 1, 2, 3,10,11,12,13),
            1: (4,5,14,15),
            2: (6,7,8,9,16,17,18,19)
        },
        'adj': empty_adj(3),
    }

    stil_631g_60 : MolConfig = {
        'name': 'STIL_631G_60',
        'xyz': stil60xyz,
        'basis': '631g',
        'ncas': [4,2,4],
        'nelecas': [4,2,4],
        'spinsub': [1, 1, 1],
        'frag_atom_list': [ [1,2,3,4,5,6,15,16,17,18,19] , [0,7, 14,20] , [8,9,10,11,12,13, 21,22,23,24,25] ],
        'frag_spin_orb': {
            0: (0, 1, 2, 3,10,11,12,13),
            1: (4,5,14,15),
            2: (6,7,8,9,16,17,18,19)
        },
        'adj': empty_adj(3),
    }

    stil_631g_90 : MolConfig = {
        'name': 'STIL_631G_90',
        'xyz': stil90xyz,
        'basis': '631g',
        'ncas': [4,2,4],
        'nelecas': [4,2,4],
        'spinsub': [1, 1, 1],
        'frag_atom_list': [ [1,2,3,4,5,6,15,16,17,18,19] , [0,7, 14,20] , [8,9,10,11,12,13, 21,22,23,24,25] ],
        'frag_spin_orb': {
            0: (0, 1, 2, 3,10,11,12,13),
            1: (4,5,14,15),
            2: (6,7,8,9,16,17,18,19)
        },
        'adj': empty_adj(3),
    }

    stil_631g_120 : MolConfig = {
        'name': 'STIL_631G_120',
        'xyz': stil120xyz,
        'basis': '631g',
        'ncas': [4,2,4],
        'nelecas': [4,2,4],
        'spinsub': [1, 1, 1],
        'frag_atom_list': [ [1,2,3,4,5,6,15,16,17,18,19] , [0,7, 14,20] , [8,9,10,11,12,13, 21,22,23,24,25] ],
        'frag_spin_orb': {
            0: (0, 1, 2, 3,10,11,12,13),
            1: (4,5,14,15),
            2: (6,7,8,9,16,17,18,19)
        },
        'adj': empty_adj(3),
    }


    stil_631g_180 : MolConfig = {
        'name': 'STIL_631G_180',
        'xyz': stil180xyz,
        'basis': '631g',
        'ncas': [4,2,4],
        'nelecas': [4,2,4],
        'spinsub': [1, 1, 1],
        'frag_atom_list': [ [1,2,3,4,5,6,15,16,17,18,19] , [0,7, 14,20] , [8,9,10,11,12,13, 21,22,23,24,25] ],
        'frag_spin_orb': {
            0: (0, 1, 2, 3,10,11,12,13),
            1: (4,5,14,15),
            2: (6,7,8,9,16,17,18,19)
        },
        'adj': empty_adj(3),
    }

    # This c4 is too small, and I observed numericall error, that H and S have all identical
    # entries however the NOCI energy is lower 
    c4_sto3g : MolConfig = {
        'name': 'C4_STO3G',
        'xyz': c4xyz,
        'basis': 'sto-3g',
        'ncas': [2,2],
        'nelecas': [2,2],
        'spinsub': [1,1],
        'frag_atom_list':  [[0,2], [3,1]],
        'frag_spin_orb': {
            0: (0,1,4,5),
            1: (2,3,6,7),
        },
        'adj': circle_adj(2), # not sure about this
    }

    c4_631g = copy.deepcopy(c4_sto3g)
    c4_631g['name'] = 'C4_631G'
    c4_631g['basis'] = '6-31g'

    c6_sto3g : MolConfig = {
        'name': 'C6_STO3G',
        'xyz': c6xyz,
        'basis': 'sto-3g',
        'ncas': [2,2,2],
        'nelecas': [2,2,2],
        'spinsub': [1,1,1],
        'frag_atom_list': [[0,2], [10,12], [3,1]], # not sure about this
        'frag_spin_orb': {
            0: (0,1,6,7),
            1: (2,3,8,9),
            2: (4,5,10,11),
        },
        'adj': circle_adj(3),
    }

    c6_631g = copy.deepcopy(c6_sto3g)
    c6_631g['name'] = 'C6_631G'
    c6_631g['basis'] = '6-31g'

    c10_631g : MolConfig = {
        'name': 'C10_631G',
        'xyz': c10xyz,
        'basis': '631g',
        'ncas': [2,2,2,2,2],
        'nelecas': [2,2,2,2,2],
        'spinsub': [1,1,1,1,1],
        'frag_atom_list': [[0,2], [10,12], [18,19], [13,11], [3,1]],
        'frag_spin_orb': {
            0: (0,1,10,11),
            1: (2,3,12,13),
            2: (4,5,14,15),
            3: (6,7,16,17),
            4: (8,9,18,19)
        },
        'adj': circle_adj(5),
    }


    h10_circle_sto3g : MolConfig = {
        'name': 'H10_CIRCLE_STO3G',
        'xyz': h10_circle_xyz,
        'basis': 'sto-3g',
        'ncas': [2,2,2,2,2],
        'nelecas': [2,2,2,2,2],
        'spinsub': [1,1,1,1,1],
        'frag_atom_list': [[0,1], [2,3], [4,5], [6,7], [8,9]],
        'frag_spin_orb': {
            0: (0,1,10,11),
            1: (2,3,12,13),
            2: (4,5,14,15),
            3: (6,7,16,17),
            4: (8,9,18,19)
        },
        'adj': circle_adj(5),
    }

    mol_configs = [stil_631g_001, stil_631g_60, stil_631g_90, stil_631g_120, stil_631g_180]

    noci_test_01 : TestConfig = {
        'epsilon': 0.01,
        'nn': False,
        'max_nx': 10000,
        'min_nc': 3,
        'init_method': 'uscc_opt',
        'frozen': 'CI',
        'grad_test': False,
        'noci_test': True
    }
        
    noci_test_001 = copy.deepcopy(noci_test_01)
    noci_test_001['epsilon'] = 0.001

    noci_test_0001 = copy.deepcopy(noci_test_01)
    noci_test_0001['epsilon'] = 0.0001

    tests = [
        noci_test_01,
        # noci_test_001,
        # noci_test_0001
    ]


    for mol_conf in mol_configs:
        print(f"Molecule {mol_conf['name']}: ")
        mol_results, ref_energy, las_energy = batch_test(mol_conf, tests)
        
        print(f"Reference energy: {ref_energy:.17f}")
        print(f"MC-LAS energy: {las_energy:.17f}")
        for result in mol_results:
            print(f"Total excitations: {result['tot_excitation_count']}")
            # print(f"NN excitations: {result['excitation_count_nn']}")

            if 'las_uscc_eng' in result:
                print(f"MC-USCC energy: {result['las_uscc_eng']:.17f}")
            if 'las_uscc_noci_eng' in result:
                print(f"MC-USCC-NOCI energy: {result['las_uscc_noci_eng']:.17f}")
            if 'tot_g' in result:
                print(f"Total gradient norm: {np.linalg.norm(result['tot_g']):.17f}")
            if 'nn_g' in result:
                print(f"NN gradient norm: {np.linalg.norm(result['nn_g']):.17f}")
            if 'las_uscc_noci_vec' in result:
                print("MC-USCC-NOCI vector: ", result['las_uscc_noci_vec'])
            if 'min_ci_eng' in result:
                print(f"Minimum CI energy: {result['min_ci_eng']:.17f}")
            print("\n")
        
        print("\n\n")
