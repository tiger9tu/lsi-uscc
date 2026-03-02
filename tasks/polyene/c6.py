import numpy as np
import pyscf
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad, lasci_ominus1
from lcc.lcc_solver import FCISolver_CC
# from lcc.las_lcc import LASSolver_LSCC
from lcc.lassi_lcc import LASSI_LSCC
from helper.util import print_list_matrix, get_sorted_excitations, cilas2f
from helper import op
from time import time
from pathlib import Path
from copy import deepcopy

pwd = Path(__file__).resolve().parent
geom_path = pwd.parent / 'geom' / 'c6.xyz'
log_path = pwd / 'data' / 'c6.log'
with geom_path.open('r') as f:
    xyz = f.read()

# Hyperparameters
#===================================
ncas_f = (2,2,2)
nelecas_f = (2,2,2)
spin_sub_f = (1,1,1)
frag_atom_list=((0, 2), (10, 11), (3, 1))
frag_sorbs = [[0,1,6,7], [2,3,8,9], [4,5,10,11]] # for jordan-wigner transformation in las_lcc.py
basis = '6-31g' 
verbose = 3

mol = gto.M (atom = xyz, basis=basis, output=log_path, verbose=verbose)
mf = scf.RHF (mol).run ()

# Running LASSCF
#===================================
las = LASSCF (mf, ncas_f, nelecas_f, spin_sub=spin_sub_f, verbose=verbose)
mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
start_time = time()
las.kernel (mo_loc)
print ("LASSCF energy = ", las.e_tot)
print("Time taken for LASSCF: {:.2f} seconds".format(time() - start_time))


cas = mcscf.CASCI (mf, sum(ncas_f), sum(nelecas_f))
cas.mo_coeff = las.mo_coeff
start_time = time()
cas.kernel ()
print ("CASCI energy = ", cas.e_tot)
print("Time taken for CASCI: {:.2f} seconds".format(time() - start_time))

all_g, g_sel, a_idxs, i_idxs = grad.get_grad_exact(las, epsilon=0.0)
gredients = np.array(g_sel)[:,0]
ordered_indices = np.argsort(-np.abs(gredients))
n_excitations = len(a_idxs)

fracs = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08]
# fracs = [0.03]
mc_uscc = mcscf.CASCI(mf, sum(ncas_f), sum(nelecas_f))
mc_uscc.mo_coeff = las.mo_coeff
h1eff, e_core = mc_uscc.get_h1eff(mc_uscc.mo_coeff)
lscc_fci = None

for frac in fracs:
    n = max(1, int(n_excitations * frac))
    a_idxs_selected = [a_idxs[i] for i in ordered_indices[:n]]
    i_idxs_selected = [i_idxs[i] for i in ordered_indices[:n]]
    print(f"\nFraction: {frac} | Number of excitations: {n}")

    tmplas = deepcopy(las)
    lsi_lscc = LASSI_LSCC(tmplas, a_idxs_selected, i_idxs_selected, frag_orbs=frag_atom_list)
    start_time = time()
    e_roots, si_rq = lsi_lscc.kernel()
    print ("LASSI-LSCC energy =", e_roots[0])
    print("Time taken: {:.2f} seconds".format(time() - start_time))
