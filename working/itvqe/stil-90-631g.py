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
xyz = ''' C    0.6125    1.4765    0.3848
C    1.7122    0.6592    0.1075
C    1.6193   -0.3700   -0.8716
C    2.6863   -1.2100   -1.1127
C    3.8573   -1.0936   -0.3748
C    3.9580   -0.1107    0.6137
C    2.9212    0.7529    0.8468
C   -0.6119    1.4761   -0.3851
C   -1.7119    0.6593   -0.1078
C   -1.6198   -0.3689    0.8740
C   -2.6870   -1.2090    1.1136
C   -3.8575   -1.0930    0.3753
C   -3.9574   -0.1113   -0.6152
C   -2.9205    0.7520   -0.8478
H    0.6927    2.1481    1.2377
H    0.6835   -0.5004   -1.3996
H    2.5990   -1.9861   -1.8614
H    4.6835   -1.7822   -0.5455
H    4.8743   -0.0269    1.1961
H    3.0071    1.5194    1.6083
H   -0.6921    2.1457   -1.2397
H   -0.6847   -0.4983    1.3997
H   -2.6003   -1.9830    1.8628
H   -4.6839   -1.7811    0.5458
H   -4.8731   -0.0284   -1.1968
H   -3.0060    1.5174   -1.6112
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



