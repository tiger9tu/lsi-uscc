# Author: Shreya Verma shreyav@uchicago.edu
# This is a sample script to run LAS-USCCSD for the H4 molecule with the polynomial-scaling algorithm to select cluster excitations
# (2e,2o)+(2e,1o)
# This is not a VQE calculation with statevector simulator, rather the classical emulator is used 

import numpy as np
import pyscf
from pyscf import gto, scf, lib, mcscf,  ao2mo
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad, lasci_ominus1
import time

def get_psi (fci, h1, h2, norb, nelec, norb_f=None, ci0_f=None,
            tol=1e-8, gtol=1e-6, max_cycle=None, 
            orbsym=None, wfnsym=None, ecore=0, opt = True, **kwargs):
    if norb_f is None: norb_f = getattr (fci, 'norb_f', [norb])
    
    if ci0_f is None: ci0_f = fci.get_init_guess (norb, nelec, norb_f, h1, h2)

    psi = getattr (fci, 'psi', fci.build_psi (ci0_f, norb, norb_f, nelec, frozen=None))
    assert (psi.check_ci0_constr)

    return psi

# Initializing the molecule with RHF
#===================================
data_dir = '/home/jinx/repo/qchem/las_uccsd_data'

with open(data_dir + '/circle/h12.xyz', 'r', encoding='utf-8') as f:
        h12_circle_xyz = f.read()
mol = gto.M (atom = h12_circle_xyz, basis = 'sto-3g', output='H12_frag_ci_test.log',verbose=0)
mf = scf.RHF (mol).run ()
# ref = mcscf.CASSCF (mf, 12, 12).run () # = FCI

# Running LASSCF
#===================================
las1 = LASSCF (mf, (2,2,2,2,2,2), (2,2,2,2,2,2), spin_sub=(1,1,1,1,1,1))
frag_atom_list1 = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11))
mo_loc1 = las1.localize_init_guess (frag_atom_list1, mf.mo_coeff)
las1_e_tot, las1_e_cas, las1_ci, las1_mo_coeff, las1_mo_energy, h2eff_sub, veff = las1.kernel (mo_loc1)


print("Shape of las1_ci:", np.shape(las1_ci))

# las2 = LASSCF (mf, (3,3,3,3), (3,3,3,3), spin_sub=(2,2,2,2))
# las2.verbose = 4
# frag_atom_list2 = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
# mo_loc2= las2.localize_init_guess (frag_atom_list2, mf.mo_coeff)
# las2.kernel (mo_loc2)

# mc_uscc_ci1 = mcscf.CASCI(mf, 12, 12)
# mc_uscc_ci1.mo_coeff = las1.mo_coeff
# mc_uscc_ci1.fcisolver = lasuccsd.FCISolver_USCC(mol, [], [])

# h1eff,e_core= mc_uscc_ci1.get_h1eff(mc_uscc_ci1.mo_coeff)
# h2eff = mc_uscc_ci1.get_h2eff()
# h = [e_core, h1eff, h2eff]

# psi1 = get_psi(fci = mc_uscc_ci1.fcisolver, h1 = h1eff, h2 = h2eff, norb = mc_uscc_ci1.ncas
#                         , nelec = mc_uscc_ci1.nelecas,  ecore = e_core, opt = False)
# eng1 = psi1.energy_tot(psi1.x,h)


# mc_uscc_ci2 = mcscf.CASCI(mf, 12, 12)
# mc_uscc_ci2.mo_coeff = las2.mo_coeff
# mc_uscc_ci2.fcisolver = lasuccsd.FCISolver_USCC(mol, [], [])
# h1eff,e_core= mc_uscc_ci2.get_h1eff(mc_uscc_ci2.mo_coeff)
# h2eff = mc_uscc_ci2.get_h2eff()
# h = [e_core, h1eff, h2eff]
# psi2 = get_psi(fci = mc_uscc_ci2.fcisolver, h1 = h1eff, h2 = h2eff, norb = mc_uscc_ci2.ncas
#                         , nelec = mc_uscc_ci2.nelecas,  ecore = e_core, opt = False)
# eng2 = psi2.energy_tot(psi2.x,h)

# print("LAS1 energy = ", las1.e_tot)
# print("LAS2 energy = ", las2.e_tot)
# # print("Reference energy = ", ref.e_tot)
# print("LAS1 CASCI energy = ", eng1)
# print("LAS2 CASCI energy = ", eng2)

# print("Las1 energy - Las2 energy = ", las1.e_tot - las2.e_tot)
# print("Las1 CASCI energy - Las2 CASCI energy = ", eng1 - eng2)

# # Get the Hamiltonian integrals in the common MO basis
# h1, ecore = las1.h1e_for_cas(mo_coeff=las1.mo_coeff)
# eri = las1.get_h2cas(las1.mo_coeff)

# # Get the CI vectors for psi1 and psi2 (assuming single-root, single-fragment for simplicity)
# ci1 = las1.ci[0][0]
# ci2 = las2.ci[0][0]

# # Use the FCI solver to compute the matrix element
# solver = las1.fciboxes[0].fcisolvers[0]
# ncas = las1.ncas_sub[0]
# nelec = las1.nelecas_sub[0]


# # h1e: one-electron integrals in MO basis
# h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff

# # eri: two-electron integrals in MO basis (chemist's notation, 4-fold symmetry)
# eri = ao2mo.kernel(mol, mf.mo_coeff)


# def _unpack_nelec(nelec, spin=None):
#     if spin is None:
#         spin = 0
#     else:
#         nelec = int(np.sum(nelec))
#     if isinstance(nelec, (int, np.number)):
#         nelecb = (nelec-spin)//2
#         neleca = nelec - nelecb
#         nelec = neleca, nelecb
#     return nelec

# def energy(fcisolver, h1e, eri, fcivec1, norb, nelec, link_index=None, fcivec2=None):
#         '''
#         Compute the FCI electronic energy for given Hamiltonian and FCI vectors.
#         If fcivec2 is provided, computes <fcivec1|H|fcivec2>, otherwise <fcivec1|H|fcivec1>.
#         '''
#         nelec = _unpack_nelec(nelec, fcisolver.spin)
#         h2e = fcisolver.absorb_h1e(h1e, eri, norb, nelec, .5)
#         if fcivec2 is None:
#                 fcivec2 = fcivec1
#         ci1 = fcisolver.contract_2e(h2e, fcivec2, norb, nelec, link_index)
#         return np.dot(fcivec1.reshape(-1), ci1.reshape(-1))

# psi1Hpsi1 = energy(solver, h1e, eri, ci1, mc_uscc_ci1.ncas, mc_uscc_ci1.nelecas) + ecore
# print("Energy <psi1|H|psi1> =", psi1Hpsi1)

# # Compute <psi1|H|psi2>
# psi1Hpsi2 = cross_energy(solver, h1e, eri, ci1, mc_uscc_ci1.ncas, mc_uscc_ci1.nelecas,  ci2) + ecore
# print("Cross energy <psi1|H|psi2> =", psi1Hpsi2)





# #Getting gradient for all cluster excitations through LAS-UCCSD gradients, may use your desired epsilon for selection
# #====================================================================================================================
# all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(las, epsilon=0.0001)
# # print ("All gradients = ", all_g)
# # print ("Selected gradients = ", g_sel)

# excitations = []
# for a, i in zip(a_idxs_selected, i_idxs_selected):
#     excitations.append((tuple(i), tuple(a[::-1])))

# # print ("Selected excitations = ", excitations)

# #Computing energy through the LAS-UCC kernel using selected excitations
# #==========================================================================================
# epsilon=0.001
# mc_uscc = mcscf.CASCI(mf, 4, 4)
# mc_uscc.mo_coeff = las.mo_coeff
# lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
# mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
# mc_uscc.fcisolver.norb_f = [2,2]
# mc_uscc.fcisolver.frozen = 'CI'
# start_time = time.time()
# mc_uscc.kernel()
# end_time = time.time()
# print("Kernel execution time: {:.6f} seconds".format(end_time - start_time))
# print("Epsilon: {:.9f} | Number of parameters: {:.0f} | LASUSCCSD energy: {:.9f}".format(epsilon, len(a_idxs_selected), mc_uscc.e_tot))
# print("ref energy = ", ref.e_tot)
# print("las energy = ", las.e_tot)




