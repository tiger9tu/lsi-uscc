import numpy as np
from argparse import ArgumentParser
from pyscf import gto, scf, lib, tools, mcscf
from mrh.my_dmet import localintegrals
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.tools import molden as molden
from mrh.exploratory.citools import fockspace
from mrh.exploratory.unitary_cc import lasuccsd as lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccs_op
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
import itertools
import pickle
from mrh.exploratory.citools import fockspace, lasci_ominus1
# from mrh.my_pyscf.mcscf import fake_rdms
# import matplotlib.pyplot as plt
from mrh.exploratory.citools import grad
from pyscf.tools import molden
# from lasvqe.las_vqe import LASVQE
# from qiskit_qulacs.qulacs_estimator import QulacsEstimator

calc_name = 'stilbene_trans'

with open("/project/lgagliardi/tuy/qchem/las-uscc-noci/working/geom/stil-90.xyz", "r") as f:
    carts = f.read()

mol = gto.M(
    atom=carts,
    basis='6-31g',
    output="stil-90-t.log",
    symmetry=False,
    charge=0,
    spin=0,
    verbose=4,
)



add_virtual_bath = True
bath_tol = 1e-8
project_cderi = False

# Hartree--Fock calculation
# --------------------------------------------------------------------------------------------------------------------

mf = scf.RHF(mol)
mf.max_cycle = 200
mf.kernel ()
molden.from_mo(mol, calc_name+'_hf.molden', mf.mo_coeff )


# Set up the localized AO basis
# --------------------------------------------------------------------------------------------------------------------
myInts = localintegrals.localintegrals(mf, range(mol.nao_nr ()), 'meta_lowdin')
#myInts.molden( my_kwargs['calcname'] + '_locints.molden' )

# Build fragments from atom list
# --------------------------------------------------------------------------------------------------------------------

frag_atom_list = [ [1,2,3,4,5,6,15,16,17,18,19] , [0,7, 14,20] , [8,9,10,11,12,13, 21,22,23,24,25] ]
frag_name_list = [ 'ph1', 'cc' , 'ph2']

# Running LASSCF using the created fraglist
# --------------------------------------------------------------------------------------------------------------------

las = LASSCF(mf, (4,2,4), (4,2,4), spin_sub=(1,3,1))
mo_coeff = las.localize_init_guess (frag_atom_list, mf.mo_coeff)

las.kernel (mo_coeff)
print("LASSCF energy triplet = ", las.e_tot)
from mrh.my_pyscf.tools import molden
molden.from_lasscf (las, 'trans-stilbene-90-t.molden')

# Running CASCI on LAS ci vector
#=====================================================
mc = mcscf.CASCI(mf, 10,10)
mc.mo_coeff = las.mo_coeff
mc.fix_spin_(ss=2)
mc.kernel()
print ("CASCI energy triplet = ", mc.e_tot)

# # # Constructing hlas using Extracting vpqrs from LAS object
# #=====================================================
# nmo = las.mo_coeff.shape[1]
# ncas, ncore = las.ncas, las.ncore
# nocc = ncore + ncas

# h0las = las.h1e_for_cas()[1]
# h1las = las.h1e_for_cas()[0]
# h2las = lib.numpy_helper.unpack_tril (las.get_h2eff().reshape (nmo*ncas,ncas*(ncas+1)//2)).reshape (nmo, ncas, ncas, ncas)[ncore:nocc,:,:,:]

# hlas = [h0las,h1las,h2las]

# # Extracting 1-,2-,3-RDMs from LAS object
# #=====================================================
# rdm1s = las.make_casdm1s()
# rdm2s = las.make_casdm2s()
# rdm3s = las.make_casdm3s()

# # Extracting single and double t amplitudes indices
# #=====================================================
# norb = las.ncas
# nlas = las.ncas_sub
# uop = lasuccsd.gen_uccsd_op(norb,nlas)
# a_idxs = uop.a_idxs
# i_idxs = uop.i_idxs

# # Calculating all gradients and then selecting
# #=====================================================
# all_g, all_gen_indices = grad.get_grad_exact(a_idxs,i_idxs,hlas, rdm1s, rdm2s, rdm3s, epsilon=0.0)
# print ("All_g = ", all_g)

# # LAS-VQE
# #=====================================================
# eps = np.array([0.0076, 0.005015, 0.00297421, 0.00252])

# lasvqe = LASVQE(mf, las, f_orbs=(4,2,4), f_elec=(4,2,4), f_atom_list=frag_atom_list, spin_sub=(1,3,1), selected=True, epsilon = eps[0])
# vqe_en, vqe_result = lasvqe.run(estimator=QulacsEstimator(),gate_counts=True, verbose=3)
# print(f"LAS-VQE Energy: {vqe_en:.12f} Ha | Epsilon: {eps[0]:.12f}")

# lasvqe = LASVQE(mf, las, f_orbs=(4,2,4), f_elec=(4,2,4), f_atom_list=frag_atom_list, spin_sub=(1,3,1), selected=True, epsilon = eps[1])
# vqe_en, vqe_result = lasvqe.run(estimator=QulacsEstimator(),gate_counts=True, verbose=3)
# print(f"LAS-VQE Energy: {vqe_en:.12f} Ha | Epsilon: {eps[1]:.12f}")

# lasvqe = LASVQE(mf, las, f_orbs=(4,2,4), f_elec=(4,2,4), f_atom_list=frag_atom_list, spin_sub=(1,3,1), selected=True, epsilon = eps[2])
# vqe_en, vqe_result = lasvqe.run(estimator=QulacsEstimator(),gate_counts=True, verbose=3)
# print(f"LAS-VQE Energy: {vqe_en:.12f} Ha | Epsilon: {eps[2]:.12f}")

# lasvqe = LASVQE(mf, las, f_orbs=(4,2,4), f_elec=(4,2,4), f_atom_list=frag_atom_list, spin_sub=(1,3,1), selected=True, epsilon = eps[3])
# vqe_en, vqe_result = lasvqe.run(estimator=QulacsEstimator(),gate_counts=True, verbose=3)
# print(f"LAS-VQE Energy: {vqe_en:.12f} Ha | Epsilon: {eps[3]:.12f}")


