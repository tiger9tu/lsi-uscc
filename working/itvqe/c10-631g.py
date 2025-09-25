# Author: Shreya Verma shreyav@uchicago.edu
# This is a sample script to run LAS-USCCSD for the H4 molecule with the polynomial-scaling algorithm to select cluster excitations
# (2e,2o)+(2e,1o)
# This is not a VQE calculation with statevector simulator, rather the classical emulator is used 

import numpy as np
import pyscf
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad, lasci_ominus1, fockspace

from typing import Iterable, Union
from scipy import linalg, optimize



def flatten(seq: Iterable) -> list:
    result = []
    for item in seq:
        if isinstance(item, (list, tuple, np.ndarray)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def my_e_de(x, h, position, psi):
    all_x = psi.x.copy()
    all_x[position] = x
    e_tot, all_jac = psi.e_de(all_x, h)
    return e_tot, all_jac[position]


def cilas2f(lasci, norb_f, nelec_f):
    ''' nelec (na, nb)'''
    ci_f = []
    for i, ci in enumerate(lasci):
        ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
    return ci_f

def get_fragment_pair_excitations(all_a_idxs, all_i_idxs, frag_pair, frag_spin_orbs):
    """
    Extract excitations within a specific fragment pair
    
    Parameters:
    -----------
    all_a_idxs, all_i_idxs : List
        All available excitation indices
    frag_pair : Tuple[int, int]
        Fragment pair indices (e.g., (0,1) for fragments 0 and 1)
        
    Returns:
    --------
    Tuple[List, List]
        Filtered excitation indices for the fragment pair
    """
    # Get combined spin orbitals for the fragment pair
    frag_orbs = set(frag_spin_orbs[frag_pair[0]] + frag_spin_orbs[frag_pair[1]])

    pair_a_idxs = []
    pair_i_idxs = []
    for idx in range(len(all_a_idxs)):
        if (
            all(elem in frag_orbs for elem in all_a_idxs[idx]) and
            all(elem in frag_orbs for elem in all_i_idxs[idx])
        ):
            pair_a_idxs.append(all_a_idxs[idx])
            pair_i_idxs.append(all_i_idxs[idx])
    return pair_a_idxs, pair_i_idxs
    


# Initializing the molecule with RHF
#===================================
xyz = '''C -5.544122 0.214299 0.000000
C 5.544122 -0.214299 0.000000
C -4.350171 -0.402736 0.000000
C 4.350171 0.402736 0.000000
H -5.622764 1.298325 0.000000
H 5.622764 -1.298325 0.000000
H -6.474236 -0.343751 0.000000
H 6.474236 0.343751 0.000000
H -4.316850 -1.492261 0.000000
H 4.316850 1.492261 0.000000
C -3.074155 0.278469 0.000000
C 3.074155 -0.278469 0.000000
C -1.872943 -0.352359 0.000000
C 1.872943 0.352359 0.000000
H -3.097453 1.368452 0.000000
H 3.097453 -1.368452 0.000000
H -1.855664 -1.442745 0.000000
H 1.855664 1.442745 0.000000
C -0.600949 0.317418 0.000000
C 0.600949 -0.317418 0.000000
H -0.616498 1.407635 0.000000
H 0.616498 -1.407635 0.000000
'''


maxiter = 10000
epsilon=0.001
ncas_f = [2,2,2,2,2]
nelec_f = ((1, 1), (1, 1), (1, 1), (1, 1),(1, 1))
spin_f = [1,1,1,1,1]
frag_atom_list = [[0,2], [10,12], [18,19], [13,11], [3,1]]


frag_spin_orb = {
            0: (0, 1, 10, 11),    # Fragment 0 spin orbitals
            1: (2, 3, 12, 13),    # Fragment 1 spin orbitals  
            2: (4, 5, 14, 15),    # Fragment 2 spin orbitals
            3: (6, 7, 16, 17),    # Fragment 3 spin orbitals
            4: (8, 9, 18, 19)     # Fragment 4 spin orbitals
        }

frag_pairs = ((0, 1), (1, 2), (2, 3), (3, 4))


ncas = sum(ncas_f)
nelec = sum(flatten(nelec_f))

mol = gto.M (atom = xyz, basis = '6-31g', output='c8_631g.log',
    verbose=0)
mf = scf.RHF (mol).run ()
ref = mcscf.CASSCF (mf, ncas, nelec).run () # = FCI
print("CASCF energy : {:.9f}".format(ref.e_tot))

# Running LASSCF
#===================================
las = LASSCF (mf, ncas_f, nelec_f, spin_sub=spin_f)


mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)

print("las energy = ", las.e_tot)


#Getting gradient for all cluster excitations through LAS-UCCSD gradients, may use your desired epsilon for selection
#====================================================================================================================

all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(las, epsilon=epsilon)




n_vqe = len(frag_pairs)

a_idxs = []
i_idxs = []

frag_pairs_sep_idxs = [0]

for frag_pair in frag_pairs:
    frag_pair_a_idxs, frag_pair_i_idxs = get_fragment_pair_excitations(a_idxs_selected, i_idxs_selected, frag_pair, frag_spin_orb)
    a_idxs.extend(frag_pair_a_idxs)
    i_idxs.extend(frag_pair_i_idxs)
    frag_pairs_sep_idxs.append(len(a_idxs))


ci0_f = cilas2f(las.ci, ncas_f, nelec_f)

frag_ci_sep_idx = []
frag_ci_sizes = [c.size for c in ci0_f]
frag_di_size = frag_ci_sizes
frag_ci_sep_idx = [sum(frag_di_size[:j]) for j in range(0, len(frag_di_size)+1)]

mc_uscc = mcscf.CASCI(mf, ncas, nelec)
mc_uscc.mo_coeff = las.mo_coeff
# lasci_ominus1.GLOBAL_MAX_CYCLE = 15
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs, i_idxs)
mc_uscc.fcisolver.norb_f = ncas_f

fci = mc_uscc.fcisolver

h1eff,e_core= mc_uscc.get_h1eff(mc_uscc.mo_coeff)
h2eff = mc_uscc.get_h2eff()
h = [e_core, h1eff, h2eff]

psi = getattr (fci, 'psi', fci.build_psi (ci0_f, ncas, ncas_f, nelec))
print("psi energy = ", psi.energy_tot(psi.x, h))

for i in range(len(frag_pairs)):
    frag_pair = frag_pairs[i]
    x_positions = [0] # constr paramter

    uop_x_start = frag_pairs_sep_idxs[i] + 1
    uop_x_end = frag_pairs_sep_idxs[i + 1] + 1
    x_positions.extend(range(uop_x_start, uop_x_end)) # uop paramter

    ci_x_start = psi.nconstr + psi.uop.ngen_uniq
    frag1_ci_x_start = ci_x_start + frag_ci_sep_idx[frag_pair[0]]
    frag1_ci_x_end = frag1_ci_x_start + ci0_f[frag_pair[0]].size

    frag2_ci_x_start = ci_x_start + frag_ci_sep_idx[frag_pair[1]]
    frag2_ci_x_end = frag2_ci_x_start + ci0_f[frag_pair[1]].size

    x_positions.extend(range(frag1_ci_x_start, frag1_ci_x_end))
    x_positions.extend(range(frag2_ci_x_start, frag2_ci_x_end))

    # print(f"Fragment pair {frag_pair} x positions: {x_positions}")
    x0 = psi.x[x_positions]

    
    res = optimize.minimize (my_e_de, x0, args=(h, x_positions, psi), method='BFGS',
    jac=True, callback=psi.get_solver_callback (h), options={'maxiter': maxiter})

    psi.x[x_positions] = res.x
    psi.converged = res.success
    print("psi iter count = ", psi.it_cnt)



e_tot = psi.energy_tot (psi.x, h)
print("Final energy after iterative VQE optimization = ", e_tot)



# compare with full vqe 
psi.x = np.zeros_like(psi.x)
# print("Reset psi.x to zero for full VQE, energy = ", psi.energy_tot(psi.x, h))
res = optimize.minimize (psi.e_de, psi.x, args=(h, ), method='BFGS',
    jac=True, callback=psi.get_solver_callback (h), options={'maxiter': maxiter})

print("psi iter count = ", psi.it_cnt)

psi.x = res.x
e_tot = psi.energy_tot (psi.x, h)
print("Final energy after full VQE optimization = ", e_tot)



