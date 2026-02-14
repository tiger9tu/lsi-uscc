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
xyz = '''H      0.000000000000   0.000000000000   0.000000000000
H      1.000000000000   0.000000000000   0.000000000000
H      0.273746762116   2.195450598147   0.100000000000
H      1.232912762116   1.895450598147  -0.100000000000'''
# Initializing the molecule with RHF
#===================================
norb = 4
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

a_idxs = np.array([[5,1]])
i_idxs = np.array([[6,0]])

#Computing energy through the LAS-UCC kernel using selected excitations
#==========================================================================================

lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs, i_idxs)
# mc_uscc.fcisolver.norb_f = ncas_f
# mc_uscc.kernel(ci0=las_ci0_f)
for a, i in zip (a_idxs, i_idxs):
    print ("a, i = ", a, i)
    errstr = 'a,i={},{} breaks sz symmetry'.format (a, i)
    #print ("SV sum orb = ",np.sum(a // norb))
    #print ("SV errstr = ", errstr)
    assert (np.sum (a//norb) == np.sum (i//norb)), errstr

print("las ci_f\n")
print_list_matrix(las_ci0_f)


psi = lasci_ominus1.LASUCCTrialState(fcisolver, las_ci0_f, norb = 4, norb_f = [2,2], nelec=[2,2])
dpci = psi.dp_ci(las_ci0_f)

print("DP-CI")
# print_list_matrix(dpci)
for i in range(dpci.shape[0]):
    for j in range(dpci.shape[1]):
        if abs(dpci[i,j]) > 1e-5:
            print(f"dpci[{i},{j}] = {dpci[i,j]:.6f}")

mc_uscc = mcscf.CASCI(mf, sum(ncas_f), sum(nelecas_f))
mc_uscc.mo_coeff = las.mo_coeff

# 
# h1eff,e_core= mc_uscc.get_h1eff(mc_uscc.mo_coeff)
# h2eff = mc_uscc.get_h2eff() 
# h = [e_core, h1eff, h2eff]
# c, uc, huc = psi.hc_x (psi.x, h)[0:3]
# uc, huc = uc.ravel (), huc.ravel ()
# cu = uc.conj ()
# cuuc = cu.dot (uc)
# cuhuc = cu.dot (huc)
# e_tot = cuhuc/cuuc
# print("c\n")
# print_sparse_ci(c.ravel(), threshold=1e-5)

# from helper.op import Op, IdentityOp, UOp, h1Op, h2Op
# assert np.allclose(c.ravel(), uc), "c and uc should be equal"
# print(f"Verification: c == uc passed")

# nmo = las.mo_coeff.shape[1]
# ncas, ncore = las.ncas, las.ncore
# nocc = ncore + ncas
# h2e =  lib.numpy_helper.unpack_tril (las.get_h2eff().reshape (nmo*ncas,ncas*(ncas+1)//2)).reshape (nmo, ncas, ncas, ncas)[ncore:nocc,:,:,:]
# h1las, h0las = las.h1e_for_cas(mo_coeff=las.mo_coeff)
# h2las = h2e

# h1lasspin = grad.get_h1e_spin(h1las)
# h2lasspin = grad.get_eri_spin(h2las)
# print_list_matrix(h2lasspin)



# print("h2lasspin[2,5,3,7] = ", h2lasspin[2,5,3,7])

# orgHop = h1Op(h1lasspin) + h2Op(h2lasspin) + e_core * IdentityOp()
# HopC = orgHop.apply(c.ravel())
# # assert np.allclose(HopC, huc.ravel()), "HopC should equal huc"
# norm_diff = np.linalg.norm(HopC - c.ravel())



# print("HopC\n")
# print_sparse_ci(HopC)
# print("huc\n")
# print_sparse_ci(huc)

# print(f"|| HopC - c || = {norm_diff:.6f}")

# now let's try a simple h to see what's the problem
# h1eff = np.zeros_like(h1eff)
# h2eff = np.zeros_like(h2eff)
# h1eff[0,1] = h1eff[1,0] = 1.0
# h = [e_core, h1eff, h2eff]
# c, uc, huc = psi.hc_x (np.zeros_like(psi.x), h)[0:3]
# uc, huc = uc.ravel (), huc.ravel ()
# cu = uc.conj ()
# cuuc = cu.dot (uc)
# cuhuc = cu.dot (huc)
# e_tot = cuhuc/cuuc
# print(f"With simple h, e_tot = {e_tot:.6f}")


# def print_sparse_ci(ci, threshold=1e-5):
#     for i in range(len(ci)):
#         if abs(ci[i]) > threshold:
#             num_digits = len(bin(len(ci) - 1)) - 2
#             print(f"ci[{i}] = {ci[i]:.6f}, i (binary) = {bin(i)[2:].zfill(num_digits)}")
# print("huc\n")
# print_sparse_ci(huc)
# print()


# print("uc\n")
# print_sparse_ci(uc)
# print()
# # simpleHOp =  get_uccsd_op(mol, a_idxs, i_idxs, norb=4, norb_f=[2,2], nelec=[2,2])
# h1spin = grad.get_h1e_spin(h1eff)
# print_list_matrix(h1spin)
# # print("h1spin\n")
# # print_list_matrix(h1spin)
# from helper.op import Op, IdentityOp, UOp, h1Op
# simpleHOp = h1Op(h1spin) + e_core * IdentityOp()
# HopC = simpleHOp.apply(c.ravel())
# print("HopC\n")
# print_sparse_ci(HopC)

# assert np.allclose(HopC, huc), "HopC should equal huc"
# print("Assertion passed: HopC == huc")

h1eff,e_core= mc_uscc.get_h1eff(mc_uscc.mo_coeff)
h2eff = mc_uscc.get_h2eff() 
# test simple h2
h1eff = np.zeros_like(h1eff)
e_core = 0.0
h2eff = np.zeros_like(h2eff)
h2eff[1, 8]  =  h2eff[8, 1] = 1.0
# (0,2) -> (1,3) excitation corresponds to h2eff[1,8] in chemist's notation
h = [e_core, h1eff, h2eff]
c, uc, huc = psi.hc_x (psi.x, h)[0:3]
uc, huc = uc.ravel (), huc.ravel ()
cu = uc.conj ()
cuuc = cu.dot (uc)
cuhuc = cu.dot (huc)
e_tot = cuhuc/cuuc
print("c\n")
print_sparse_ci(c.ravel(), threshold=1e-5)

print("huc\n")
print_sparse_ci(huc)

from helper.op import Op, IdentityOp, UOp, h1Op, h2Op
assert np.allclose(c.ravel(), uc), "c and uc should be equal"
print(f"Verification: c == uc passed")

nmo = las.mo_coeff.shape[1]
ncas, ncore = las.ncas, las.ncore
nocc = ncore + ncas


h2efflas = las.get_h2eff() #.reshape (nmo*ncas,ncas*(ncas+1)//2)

# (nmo, ncas*ncas*(ncas+1)//2)
h2e =  lib.numpy_helper.unpack_tril (las.get_h2eff().reshape (nmo*ncas,ncas*(ncas+1)//2)).reshape (nmo, ncas, ncas, ncas)[ncore:nocc,:,:,:]
h2e = np.zeros_like(h2e)
h2e[3,2,0,1] = 1.0
# h2e[3,1,2,0] = 1.0
h1las, h0las = las.h1e_for_cas(mo_coeff=las.mo_coeff)
h2las = h2e

h1lasspin = grad.get_h1e_spin(h1las)
h2lasspin = grad.get_eri_spin(h2las)

# q,s,r,p -> p,q,r,s
# h2lasspin_conventional = h2lasspin.transpose(3,0,2,1)
for i in range(h2lasspin.shape[0]):
    for j in range(h2lasspin.shape[1]):
        for k in range(h2lasspin.shape[2]):
            for l in range(h2lasspin.shape[3]):
                if abs(h2lasspin[i,j,k,l]) > 1e-5:
                    print(f"h2lasspin[{i},{j},{k},{l}] = {h2lasspin[i,j,k,l]:.6f}")

h2lasspin = h2lasspin.transpose(3, 0, 2, 1)
for i in range(h2lasspin.shape[0]):
    for j in range(h2lasspin.shape[1]):
        for k in range(h2lasspin.shape[2]):
            for l in range(h2lasspin.shape[3]):
                if abs(h2lasspin[i,j,k,l]) > 1e-5:
                    print(f"h2lasspinTransposepqrs[{i},{j},{k},{l}] = {h2lasspin[i,j,k,l]:.6f}")
# h1lasspin = np.zeros_like(h1lasspin)
# h2lasspin = np.zeros_like(h2lasspin)
# print_list_matrix(h2lasspin)
# Hop = h1Op(h1lasspin) + h2Op(h2lasspin) + e_core * IdentityOp()
Hop = 0.5 * h2Op(h2lasspin)
HopC = Hop.apply(c.ravel())
print("HopC\n")
print_sparse_ci(HopC)