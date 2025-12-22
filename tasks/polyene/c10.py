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
geom_path = pwd.parent / 'geom' / 'c10.xyz'
log_path = pwd / 'data' / 'c10.log'
with geom_path.open('r') as f:
    xyz = f.read()

# Hyperparameters
#===================================
ncas_f = (2,2,2,2,2)
nelecas_f = (2,2,2,2,2)
spin_sub_f = (1,1,1,1,1)
frag_atom_list=[[0,2], [10,12], [18,19], [13,11], [3,1]]
basis = '6-31g' 
verbose = 3

mol = gto.M (atom = xyz, basis=basis, output=log_path, verbose=verbose)
mf = scf.RHF (mol).run ()

# Running LASSCF
#===================================
las = LASSCF (mf, ncas_f, nelecas_f, spin_sub=spin_sub_f, verbose=verbose)
mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)
if las.converged is not True:
    print("LASSCF did not converge!")
print ("LASSCF energy = ", las.e_tot)
las_ci0_f = cilas2f(las.ci, ncas_f, nelecas_f)


cas = mcscf.CASCI (mf, sum(ncas_f), sum(nelecas_f))
cas.mo_coeff = las.mo_coeff
cas.kernel ()
if cas.converged is not True:
    print("CASCI did not converge!")
print ("CASCI energy = ", cas.e_tot)


all_g, g_sel, a_idxs, i_idxs = grad.get_grad_exact(las, epsilon=0.0)
gredients = np.array(g_sel)[:,0]
ordered_indices = np.argsort(-np.abs(gredients))
n_excitations = len(a_idxs)

fracs = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08]
mc_uscc = mcscf.CASCI(mf, sum(ncas_f), sum(nelecas_f))
mc_uscc.mo_coeff = las.mo_coeff

lscc_fci = None

for frac in fracs:
    n = max(1, int(n_excitations * frac))
    a_idxs_selected = [a_idxs[i] for i in ordered_indices[:n]]
    i_idxs_selected = [i_idxs[i] for i in ordered_indices[:n]]
    print(f"\nFraction: {frac} | Number of excitations: {n}")
    #Computing energy through the LAS-UCC kernel using selected excitations
    #==========================================================================================

    lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
    mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
    mc_uscc.fcisolver.norb_f = ncas_f
    mc_uscc.kernel(ci0=las_ci0_f)
    print("LASUSCCSD-VQE energy: {:.9f}".format(mc_uscc.e_tot))
    if mc_uscc.converged is not True:
        print("LASUSCCSD-VQE did not converge!")
    a_idxs_selected.insert(0, np.array([0], dtype=np.uint8))
    i_idxs_selected.insert(0, np.array([0], dtype=np.uint8))
    # print("a_idxs_selected = ", a_idxs_selected)
    # t does not matter now, just to avoid error
    if lscc_fci is None:
        lscc_fci = FCISolver_CC(mol, a_idxs_selected, i_idxs_selected)
    else:
        lscc_fci.a_idxs = a_idxs_selected
        lscc_fci.i_idxs = i_idxs_selected

    mc_uscc.fcisolver = lscc_fci

    mc_uscc.fcisolver.norb_f = ncas_f
    mc_uscc.kernel(ci0=las_ci0_f)
    print("LASUSCCSD-CC energy: {:.9f}".format(mc_uscc.e_tot))
