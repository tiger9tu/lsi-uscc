import numpy as np
import pyscf
import os
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad, lasci_ominus1
from helper.util import print_list_matrix, get_sorted_excitations, cilas2f
from lcc.lcc_solver import FCISolver_CC
# specify the hyperparameters


ncas_f = (2,2)
nelecas_f = (2,2)
spin_sub_f = (1,1)
frag_atom_list=((0, 2), (3, 1))
fraction = 0.04

print("hyperparameters:")
print(f"ncas_f = {ncas_f}")
print(f"nelecas_f = {nelecas_f}")
print(f"spin_sub_f = {spin_sub_f}")
print(f"frag_atom_list = {frag_atom_list}")
print(f"excitation fraction = {fraction}")

# Initializing the molecule with RHF
#===================================


pwd = os.path.dirname(os.path.abspath(__file__))
with open(f'{pwd}/../geom/c4.xyz', 'r', encoding='utf-8') as f:
    c4xyz = f.read()

mol = gto.M (atom = c4xyz, basis = '6-31g', output='c4_6-31g.log', verbose=0)
mf = scf.RHF (mol).run ()
# ref = mcscf.CASCI (mf, sum(ncas_f), sum(nelecas_f)).run () 
print ("RHF energy = ", mf.e_tot)
# print ("CASCI energy = ", ref.e_tot)

# Running LASSCF
#===================================
las = LASSCF (mf, ncas_f, nelecas_f, spin_sub=spin_sub_f)
las.verbose = 4
mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)
print ("LASSCF energy = ", las.e_tot)
las_ci0_f = cilas2f(las.ci, ncas_f, nelecas_f)

# Run casci
mc_casci = mcscf.CASCI(mf, sum(ncas_f), sum(nelecas_f))
mc_casci.mo_coeff = las.mo_coeff
mc_casci.kernel()
print("CASCI energy: {:.9f}".format(mc_casci.e_tot))

#Getting gradient for all cluster excitations through LAS-UCCSD gradients, may use your desired epsilon for selection
#====================================================================================================================
a_idxs_selected, i_idxs_selected = get_sorted_excitations(las, fraction=fraction)
excitations = []
for idx,(a, i) in enumerate(zip(a_idxs_selected, i_idxs_selected)):
    print(f"Selected excitation {idx}: a = {a}, i = {i}")


#Computing energy through the LAS-UCC kernel using selected excitations
#==========================================================================================
mc_uscc = mcscf.CASCI(mf, sum(ncas_f), sum(nelecas_f))
mc_uscc.mo_coeff = las.mo_coeff
lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
mc_uscc.fcisolver.norb_f = ncas_f
mc_uscc.kernel(ci0=las_ci0_f)
print("LASUSCCSD energy: {:.9f}".format(mc_uscc.e_tot))
vqe_psi = mc_uscc.fcisolver.psi
print("final vqe amplitudes:")
print_list_matrix(vqe_psi.x[vqe_psi.nconstr:vqe_psi.nconstr + len(a_idxs_selected)])

# include identity operator, notice that we have to insert it at front otherwise it might get filtered out
a_idxs_selected.insert(0, np.array([0], dtype=np.uint8))
i_idxs_selected.insert(0, np.array([0], dtype=np.uint8))

mc_uscc.fcisolver = FCISolver_CC(mol, a_idxs_selected, i_idxs_selected, t = 1000)
mc_uscc.fcisolver.norb_f = ncas_f
mc_uscc.kernel(ci0=las_ci0_f, verbose=4)
print("LASLCCSD energy: {:.9f}".format(mc_uscc.e_tot))
print("final lcc amplitudes:")
print_list_matrix(mc_uscc.fcisolver.si)