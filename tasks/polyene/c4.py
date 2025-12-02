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
geom_path = pwd.parent / 'geom' / 'c4.xyz'
log_path = pwd / 'data' / 'c4.log'
with geom_path.open('r') as f:
    xyz = f.read()

# Hyperparameters
#===================================
ncas_f = (2,2)
nelecas_f = (2,2)
spin_sub_f = (1,1)
frag_atom_list=((0, 2), (3, 1))
basis = '6-31g' 
# epsilon = 0.001
fraction = 0.01
verbose = 3


mol = gto.M (atom = xyz, basis=basis, output=log_path, verbose=verbose)
mf = scf.RHF (mol).run ()
ref = mcscf.CASCI (mf, sum(ncas_f), sum(nelecas_f)).run () # = FCI
print ("RHF energy = ", mf.e_tot)
print ("CASCI energy = ", ref.e_tot)


# Running LASSCF
#===================================
las = LASSCF (mf, ncas_f, nelecas_f, spin_sub=spin_sub_f, verbose=verbose)
mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)
print ("LASSCF energy = ", las.e_tot)


# all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(las, epsilon)
a_idxs_selected, i_idxs_selected = get_sorted_excitations(las, fraction=fraction, verbose=verbose)

#Computing energy through the LAS-UCC kernel using selected excitations
#==========================================================================================
mc_uscc = mcscf.CASCI(mf, sum(ncas_f), sum(nelecas_f))
mc_uscc.mo_coeff = las.mo_coeff
lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
mc_uscc.fcisolver.norb_f = ncas_f
mc_uscc.kernel()
print("Fraction: {:.9f} | Number of parameters: {:.0f} | LASUSCCSD-VQE energy: {:.9f}".format(fraction, len(a_idxs_selected), mc_uscc.e_tot))


a_idxs_selected.insert(0, np.array([0], dtype=np.uint8))
i_idxs_selected.insert(0, np.array([0], dtype=np.uint8))
# print("a_idxs_selected = ", a_idxs_selected)
# t does not matter now, just to avoid error
mc_uscc.fcisolver = FCISolver_CC(mol, a_idxs_selected, i_idxs_selected, t = 1000)

mc_uscc.fcisolver.norb_f = ncas_f
las_ci0_f = cilas2f(las.ci, ncas_f, nelecas_f)
# Print numpy dtype if array-like, and also Python type of the first element
mc_uscc.kernel(ci0=las_ci0_f)
print("LASLCCSD energy: {:.9f}".format(mc_uscc.e_tot))