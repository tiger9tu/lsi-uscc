# test the distribution of the nearest neighbor excitations

import numpy as np
import pyscf
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad, lasci_ominus1

from typing import Iterable, Union
from typing import Dict, List, Any, TypedDict
import copy
import pickle
from enum import Enum
from scipy.linalg import eigh
from scipy import optimize

def flatten(seq: Iterable) -> list:
    result = []
    for item in seq:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def is_nn(idxes, config):
    spin_orbs = flatten(idxes)
    # Find and print the fragment for each spin orbital in the excitation
    fragments = []
    for orb in spin_orbs:
        frag = [k for k, v in config['frag_spin_orb'].items() if orb in v]
        if(frag and frag[0] not in fragments):
            fragments.append(frag[0])
        if not frag:
            print("Warning: belongs to no fragments, orb = ", orb)
    return len(fragments) <= 2 and config['adj'][fragments[0]][fragments[1]] == 1

GLOBAL_MAX_CYCLE = 15000 # debug 


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
    init_method: str # 'random' or 'uscc_opt'
    max_nx: int  # max number of excitations per configuration
    min_nc: int
    # test choices
    grad_test: bool
    noci_test: bool  

class TestResult(TypedDict):
    tot_g : List[float]
    nn_g : List[float]
    tot_excitation_count: int
    excitation_count_nn: int
    las_uscc_energy: float
    las_uscc_noci_energy: float

def get_Sij_Hij(psi_i, psi_j, h):
    ucj, hucj = psi_j.hc_x (psi_j.x, h)[1:3]
    uci, huci = psi_i.hc_x (psi_i.x, h)[1:3]
    ucj, hucj = ucj.ravel (), hucj.ravel ()
    uci = uci.ravel ()
    Sij = uci.conj ().dot (ucj)
    Hij = uci.conj ().dot (hucj)
    return Sij, Hij


def psi_kernel (fci, h1, h2, norb, nelec, norb_f=None, ci0_f=None,
            tol=1e-8, gtol=1e-6, max_cycle=None, 
            orbsym=None, wfnsym=None, ecore=0, opt = True, **kwargs):
    if norb_f is None: norb_f = getattr (fci, 'norb_f', [norb])
    
    if ci0_f is None: ci0_f = fci.get_init_guess (norb, nelec, norb_f, h1, h2)

    psi = getattr (fci, 'psi', fci.build_psi (ci0_f, norb, norb_f, nelec, frozen=None))
    assert (psi.check_ci0_constr)

    # debug
    #print("len(psi.x) = ", len(psi.x))
    #print("len(psi.uop.ngen_uniq) = ", psi.uop.ngen_uniq)
    #print("sum ([c.size for c in psi.ci_f] = ", sum ([c.size for c in psi.ci_f]))
    
    if not opt:
        return psi 
    

    if max_cycle is None:
        max_cycle = GLOBAL_MAX_CYCLE if GLOBAL_MAX_CYCLE is not None else 15000
    psi_options = {'gtol':     gtol,
                   'maxiter':  max_cycle}
    

    h = [ecore, h1, h2]
    psi_callback = psi.get_solver_callback (h)
    
    res = optimize.minimize (psi.e_de, psi.x, args=(h,), method='BFGS',
        jac=True, callback=psi_callback, options=psi_options)
    if not res.success:
        print('Warning: optimization failed, res.message = ', res.message)

    psi.x = res.x
    psi.converged = res.success
    psi.finalize_()
    return psi



def get_nn_excitations(a_idxs_selected, i_idxs_selected, all_g, config):
    nn_g = []
    a_idxs_selected_nn = []
    i_idxs_selected_nn = []
    
    for idx, (a, i) in enumerate((zip(a_idxs_selected, i_idxs_selected))):
        excitation = (tuple(i), tuple(a[::-1]))
        if(is_nn(excitation, config)):
            nn_g.append(all_g[idx])
            a_idxs_selected_nn.append(a_idxs_selected[idx])
            i_idxs_selected_nn.append(i_idxs_selected[idx])
    
    return nn_g, a_idxs_selected_nn, i_idxs_selected_nn


def arr_split(arr, n):
    """Split an array into n nearly equal parts."""
    k, m = divmod(len(arr), n)
    return list(arr[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def print_matrix(mat):
    for row in mat:
        print("  ".join(f"{x:.12f}" for x in row))

def nci_test(a_idxs_selected, i_idxs_selected, config, mol, mc_uscc):

    nx = len(a_idxs_selected)
    nc = nx // config['max_nx']
    if nc < config['min_nc']:
        nc = config['min_nc']
    h1eff,e_core= mc_uscc.get_h1eff(mc_uscc.mo_coeff)
    h2eff = mc_uscc.get_h2eff()
    #print("a_idxs_selected = ", a_idxs_selected)
    a_splits = arr_split(a_idxs_selected, nc - 1)
    i_splits = arr_split(i_idxs_selected, nc - 1)
    a_splits.append(np.array([], dtype=int)) 
    i_splits.append(np.array([], dtype=int))  
    las_ucc_trial_cis = []

    for i in range(nc):
        mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_splits[i], i_splits[i])
        psi = None
        if(config['init_method'] == 'random'):
            psi = psi_kernel(fci = mc_uscc.fcisolver, h1 = h1eff, h2 = h2eff, norb = mc_uscc.ncas
                               , nelec = mc_uscc.nelecas,  ecore = e_core, opt = False)
            len_psi_xcc = psi.uop.ngen_uniq
            rand_xcc_var = np.random.rand(len_psi_xcc)
            rand_xcc_var /= np.linalg.norm(rand_xcc_var)
            psi.x[psi.nconstr:psi.uop.ngen_uniq + psi.nconstr] = rand_xcc_var 
        elif(config['init_method'] == 'uscc_opt'):
            psi = psi_kernel(fci = mc_uscc.fcisolver, h1 = h1eff, h2 = h2eff, norb = mc_uscc.ncas
                                         , nelec = mc_uscc.nelecas,  ecore = e_core)
        else:
            raise ValueError("init_method must be 'random' or 'uscc_opt'")
      
        las_ucc_trial_cis.append(psi)
    
    S = np.zeros((nc, nc), dtype=np.complex128)
    H = np.zeros((nc, nc), dtype=np.complex128)
    h = [e_core, h1eff, h2eff]

    for i in range(nc):
        for j in range(nc):
            S[i, j], H[i, j] = get_Sij_Hij(las_ucc_trial_cis[i], las_ucc_trial_cis[j], h)

    print("S matrix:")
    print_matrix(S)
    print("H matrix:")
    print_matrix(H)
    eigvals, eigvecs = eigh(H, S)
    return eigvals[0]  # Take the lowest eigenvalue as the energy



def test(mol_config, test_config,las, mc_uscc, mol):
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
        
    mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
    mc_uscc.fcisolver.norb_f = mol_config['ncas'] # number of orbitals in each fragment
    # easily hit the maximal memory limit
    mc_uscc.kernel()
    if not mc_uscc.converged:
        print('Warning: kernel hasn\'t converged')
    
    result['las_uscc_eng'] =  mc_uscc.e_tot 
   
    if test_config['noci_test']:
        result['las_uscc_noci_eng'] = nci_test(a_idxs_selected, i_idxs_selected, test_config, mol, mc_uscc)
    
    return result
   



def batch_test(mol_config, test_configs):
    results = []

    # Initializing the molecule with RHF
    #===================================
    mol = gto.M(atom=mol_config['xyz'], basis=mol_config['basis'], verbose=0,output=None)
    mf = scf.RHF(mol).run()

    # Running LASSCF
    #===================================
    las = LASSCF(mf, mol_config['ncas'], mol_config['nelecas'], spin_sub=mol_config['spinsub'], ouput=None)
    mo_loc = las.localize_init_guess(mol_config['frag_atom_list'], mf.mo_coeff)
    las.kernel(mo_loc)
    
    ref = mcscf.CASSCF(mf, sum(mol_config['ncas']), sum(mol_config['nelecas'])).run() # = FCI

    mc_uscc = mcscf.CASCI(mf, sum(mol_config['ncas']), sum(mol_config['nelecas']))
    mc_uscc.mo_coeff = las.mo_coeff
    for test_config in test_configs:
        
        result = test(mol_config, test_config,las, mc_uscc, mol)
        results.append(result)
    
    return results, ref.e_tot, las.e_tot


def empty_adj(n):
    return [[0 for _ in range(n)] for _ in range(n)]

def circle_adj(n):
    return [[1 if abs(i - j) == 1 or abs(i - j) == n-1 else 0 for j in range(n)] for i in range(n)]


if __name__ == "__main__":
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

    data_dir = '/home/tuy/repo/las_uccsd_data'

    with open(data_dir + '/stilbene/geometries/stil-90.xyz', 'r', encoding='utf-8') as f:
        stil90xyz = f.read()
    
    with open(data_dir + '/polyenes/geometries/c4.xyz', 'r', encoding='utf-8') as f:
        c4xyz = f.read()

    with open(data_dir + '/polyenes/geometries/c6.xyz', 'r', encoding='utf-8') as f:
        c6xyz = f.read()

    with open(data_dir + '/polyenes/geometries/c10.xyz', 'r', encoding='utf-8') as f:
        c10xyz = f.read()

    with open(data_dir + '/circle/H10.xyz', 'r', encoding='utf-8') as f:
        h10_circle_xyz = f.read()
    
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

    stil_sto3g_90 : MolConfig = {
        'name': 'STIL_STO3G_90',
        'xyz': stil90xyz,
        'basis': 'sto-3g',
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

    c10_sto3g : MolConfig = {
        'name': 'C10_STO3G',
        'xyz': c10xyz,
        'basis': 'sto-3g',
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

    mol_configs = [c4_sto3g, c4_631g, c6_631g, h10_circle_sto3g, stil_sto3g_90, c10_sto3g ]

    noci_test_01 : TestConfig = {
        'epsilon': 0.01,
        'nn': False,
        'max_nx': 10000,
        'min_nc': 3,
        'init_method': 'uscc_opt',
        'grad_test': False,
        'noci_test': True
    }
        
    noci_test_001 = copy.deepcopy(noci_test_01)
    noci_test_001['epsilon'] = 0.001

    noci_test_0001 = copy.deepcopy(noci_test_01)
    noci_test_0001['epsilon'] = 0.0001

    tests = [
        noci_test_01,
        noci_test_001,
        noci_test_0001
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
            print("\n")
        
        print("\n\n")

