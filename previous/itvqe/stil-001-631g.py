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
xyz = ''' C      -0.66913973      1.85730726     -0.03805098
C      -1.61065036      0.73885071     -0.09574696
C      -1.35340246     -0.43288283     -0.81483568
C      -2.28594009     -1.45288042     -0.86682865
C      -3.49986561     -1.32918247     -0.20406092
C      -3.77981149     -0.16564793      0.49591780
C      -2.85072586      0.86034066      0.53755379
C       0.66914373      1.85730826      0.03787198
C       1.61065936      0.73880971      0.09552496
C       1.35328446     -0.43291083      0.81460268
C       2.28583609     -1.45287842      0.86681365
C       3.49987661     -1.32917547      0.20424692
C       3.77992149     -0.16567693     -0.49573780
C       2.85081786      0.86029966     -0.53757779
H      -1.14269154      2.83473987     -0.06049798
H      -0.41631383     -0.53217479     -1.34747946
H      -2.06882318     -2.34798206     -1.43597143
H      -4.22727632     -2.12958615     -0.24491490
H      -4.72859812     -0.05278198      1.00495860
H      -3.07755377      1.77276929      1.07689257
H       1.14262454      2.83473387      0.06205798
H       0.41611283     -0.53215779      1.34710046
H       2.06866718     -2.34795906      1.43596843
H       4.22730132     -2.12955815      0.24528090
H       4.72878912     -0.05283198     -1.00463060
H       3.07771977      1.77270129     -1.07692857
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



