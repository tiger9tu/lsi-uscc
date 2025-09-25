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
xyz = ''' C 0.637297 1.609894 0.271644
C 1.671915 0.637125 0.095037
C 1.555787 -0.429383 -0.818561
C 2.601892 -1.301075 -1.027366
C 3.804515 -1.141541 -0.343657
C 3.944438 -0.092275 0.554759
C 2.896357 0.779603 0.774827
C -0.637289 1.610117 -0.271666
C -1.671947 0.637404 -0.094976
C -1.556248 -0.428352 0.819552
C -2.602303 -1.300124 1.028279
C -3.804491 -1.141341 0.343632
C -3.944028 -0.092767 -0.555655
C -2.895976 0.77916 -0.775661
H 0.902253 2.448365 0.916227
H 0.632452 -0.54494 -1.373252
H 2.490883 -2.108975 -1.73971
H 4.627795 -1.820335 -0.523243
H 4.877985 0.045 1.08573
H 3.011386 1.601166 1.472498
H -0.901646 2.447808 -0.917488
H -0.633279 -0.543245 1.374995
H -2.491625 -2.107457 1.741317
H -4.627761 -1.820157 0.52318
H -4.877252 0.043926 -1.087344
H -3.010699 1.600174 -1.474031
'''



maxiter = 10000
epsilon=0.001
ncas_f = (4, 2, 4)
nelec_f = ((2, 2), (1, 1), (2, 2))
spin_f = [1,1,1]
frag_atom_list = ((1,2,3,4,5,6,15,16,17,18,19), (0,7,14,20), (8,9,10,11,12,13,21,22,23,24,25))




frag_spin_orb = {
            0: (0, 1, 2, 3, 10, 11, 12, 13),    # Fragment 0: Phenyl ring 1
            1: (4, 5, 14, 15),                  # Fragment 1: Vinyl bridge
            2: (6, 7, 8, 9, 16, 17, 18, 19)    # Fragment 2: Phenyl ring 2
        }
frag_pairs = ((0, 1), (1, 2),(0,2))  # Fragment pairs for NOCI


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



