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
def print_sparse_ci(ci, threshold=1e-5):
    for i in range(len(ci)):
        if abs(ci[i]) > threshold:
            num_digits = len(bin(len(ci) - 1)) - 2
            print(f"ci[{i}] = {ci[i]:.6f}, i (binary) = {bin(i)[2:].zfill(num_digits)}")
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
print ("RHF energy = ", mf.e_tot)

# Running LASSCF
#===================================
las = LASSCF (mf, ncas_f, nelecas_f, spin_sub=spin_sub_f, verbose=3)
mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)
print ("LASSCF energy = ", las.e_tot)
las_ci0_f = cilas2f(las.ci, ncas_f, nelecas_f)


cas = mcscf.CASCI (mf, sum(ncas_f), sum(nelecas_f))
cas.mo_coeff = las.mo_coeff
cas.kernel ()
print ("CASCI energy = ", cas.e_tot)


mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)
print ("LASSCF energy = ", las.e_tot)
las_ci0_f = cilas2f(las.ci, ncas_f, nelecas_f)
#Getting gradient for all cluster excitations through LAS-UCCSD gradients, may use your desired epsilon for selection
#====================================================================================================================

mc_uscc = mcscf.CASCI(mf, sum(ncas_f), sum(nelecas_f))
mc_uscc.mo_coeff = las.mo_coeff
h1eff,e_core = mc_uscc.get_h1eff(mc_uscc.mo_coeff)
h2eff = mc_uscc.get_h2eff() 
# print("h2eff = ", h2eff)
print_list_matrix(h2eff, digits=3)

nmo = las.mo_coeff.shape[1]
ncas, ncore = las.ncas, las.ncore
nocc = ncore + ncas
h2e = lib.numpy_helper.unpack_tril (las.get_h2eff().reshape (nmo*ncas,ncas*(ncas+1)//2)).reshape (nmo, ncas, ncas, ncas)[ncore:nocc,:,:,:]
h1las, h0las = las.h1e_for_cas(mo_coeff=las.mo_coeff)
h2las = h2e
h2lasspin = grad.get_eri_spin(h2las).transpose(3,0,2,1)
print("h2las[0,1,2,3] = ", h2las[0,1,2,3]) # should = h2eff[8,1]


# print the hc
h1eff,e_core= mc_uscc.get_h1eff(mc_uscc.mo_coeff)
h2eff = mc_uscc.get_h2eff() 
h = [e_core, h1eff, h2eff]
a_idxs = np.array([[5,1]])
i_idxs = np.array([[6,0]])
fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs, i_idxs)
psi = lasci_ominus1.LASUCCTrialState(fcisolver, las_ci0_f, norb = 4, norb_f = [2,2], nelec=[2,2])
c, uc, huc = psi.hc_x (psi.x, h)[0:3]
uc, huc = uc.ravel (), huc.ravel ()
print("c\n")
print_sparse_ci(c.ravel(), threshold=1e-5)
print("huc\n")
print_sparse_ci(huc)


h2e =  lib.numpy_helper.unpack_tril (las.get_h2eff().reshape (nmo*ncas,ncas*(ncas+1)//2)).reshape (nmo, ncas, ncas, ncas)[ncore:nocc,:,:,:]
h1las, h0las = las.h1e_for_cas(mo_coeff=las.mo_coeff)
h2las = h2e

h1lasspin = grad.get_h1e_spin(h1las)
h2lasspin = grad.get_eri_spin(h2las).transpose(3,0,2,1)

from helper.op import Op, IdentityOp, UOp, h1Op, h2Op
Hop = h1Op(h1lasspin) + 0.5 * h2Op(h2lasspin) + e_core * IdentityOp()
HopC = Hop.apply(c.ravel())
print("HopC\n")
print_sparse_ci(HopC)

assert np.allclose(HopC, huc), "c and uc should be equal"

# all_g, g_sel, a_idxs, i_idxs = grad.get_grad_exact(las, epsilon=0.0)
# gredients = np.array(g_sel)[:,0]
# ordered_indices = np.argsort(-np.abs(gredients))
# n_excitations = len(a_idxs)

# fracs = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08]
# mc_uscc = mcscf.CASCI(mf, sum(ncas_f), sum(nelecas_f))
# mc_uscc.mo_coeff = las.mo_coeff

# lscc_fci = None

# for frac in fracs:
#     n = max(1, int(n_excitations * frac))
#     a_idxs_selected = [a_idxs[i] for i in ordered_indices[:n]]
#     i_idxs_selected = [i_idxs[i] for i in ordered_indices[:n]]
#     print(f"\nFraction: {frac} | Number of excitations: {n}")
#     #Computing energy through the LAS-UCC kernel using selected excitations
#     #==========================================================================================

#     lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
#     mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
#     mc_uscc.fcisolver.norb_f = ncas_f
#     mc_uscc.kernel(ci0=las_ci0_f)
#     print("LASUSCCSD-VQE energy: {:.9f}".format(mc_uscc.e_tot))
#     a_idxs_selected.insert(0, np.array([0], dtype=np.uint8))
#     i_idxs_selected.insert(0, np.array([0], dtype=np.uint8))
#     # print("a_idxs_selected = ", a_idxs_selected)
#     # t does not matter now, just to avoid error
#     if lscc_fci is None:
#         lscc_fci = FCISolver_CC(mol, a_idxs_selected, i_idxs_selected)
#     else:
#         lscc_fci.a_idxs = a_idxs_selected
#         lscc_fci.i_idxs = i_idxs_selected

#     mc_uscc.fcisolver = lscc_fci

#     mc_uscc.fcisolver.norb_f = ncas_f
#     mc_uscc.kernel(ci0=las_ci0_f)
#     print("LASUSCCSD-CC energy: {:.9f}".format(mc_uscc.e_tot))