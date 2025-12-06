import numpy as np
import pyscf
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad, lasci_ominus1
from lcc.lcc_solver import FCISolver_CC
from helper.util import print_list_matrix, get_sorted_excitations, cilas2f

from pathlib import Path

pwd = Path(__file__).resolve().parent
geom_path = pwd.parent / 'geom' / 'h4.xyz'
with geom_path.open('r') as f:
    xyz = f.read()

# Initializing the molecule with RHF
#===================================
ncas_f = (2,2)
nelecas_f = (2,2)
spin_sub_f = (1,1)
frag_atom_list = ((0,1),(2,3))

mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log',
    verbose=0)
mf = scf.RHF (mol).run ()
ref = mcscf.CASCI (mf, 4, 4).run () # = FCI
print ("RHF energy = ", mf.e_tot)
print ("CASCI energy = ", ref.e_tot)


# Running LASSCF
#===================================
las = LASSCF (mf, ncas_f, nelecas_f, spin_sub=spin_sub_f)
las.verbose = 4

mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)
print ("LASSCF energy = ", las.e_tot)
las_ci0_f = cilas2f(las.ci, ncas_f, nelecas_f)
#Getting gradient for all cluster excitations through LAS-UCCSD gradients, may use your desired epsilon for selection
#====================================================================================================================

epsilon = 0.001
all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(las, epsilon)
# sort the selected excitations by gradiDent magnitude
gredients = np.array(g_sel)[:,0]
sorted_indices = np.argsort(-np.abs(gredients))
a_idxs_selected = [a_idxs_selected[i] for i in sorted_indices]
i_idxs_selected = [i_idxs_selected[i] for i in sorted_indices]


#Computing energy through the LAS-UCC kernel using selected excitations
#==========================================================================================
epsilon=0.01
mc_uscc = mcscf.CASCI(mf, sum(ncas_f), sum(nelecas_f))
mc_uscc.mo_coeff = las.mo_coeff
lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
mc_uscc.fcisolver.norb_f = [2,2]
mc_uscc.kernel(ci0=las_ci0_f)
print("Epsilon: {:.9f} | Number of parameters: {:.0f} | LASUSCCSD energy: {:.9f}".format(epsilon, len(a_idxs_selected), mc_uscc.e_tot))

# print("a_idxs_selected = ", a_idxs_selected)
# to include the Identity operator
# a_idxs_selected = []
# i_idxs_selected = []
a_idxs_selected.insert(0, np.array([0], dtype=np.uint8))
i_idxs_selected.insert(0, np.array([0], dtype=np.uint8))
# print("a_idxs_selected = ", a_idxs_selected)
# t does not matter now, just to avoid error
mc_uscc.fcisolver = FCISolver_CC(mol, a_idxs_selected, i_idxs_selected, t = 1000)

mc_uscc.fcisolver.norb_f = ncas_f

mc_uscc.kernel(ci0=las_ci0_f)
print("LASLCCSD energy: {:.9f}".format(mc_uscc.e_tot))

# c = mc_uscc.fcisolver.psi0.dp_ci (las_ci0_f)
# print("norm c = ", np.linalg.norm(c))