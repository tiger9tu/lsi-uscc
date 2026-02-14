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

all_g, g_sel, a_idxs, i_idxs = grad.get_grad_exact(las, epsilon=0.0)
gredients = np.array(g_sel)[:,0]
ordered_indices = np.argsort(-np.abs(gredients))
n_excitations = len(a_idxs)

# fracs = [0.04]
mc_uscc = mcscf.CASCI(mf, sum(ncas_f), sum(nelecas_f))
mc_uscc.mo_coeff = las.mo_coeff
n = 10
a_idxs_selected = [a_idxs[i] for i in ordered_indices[:n]]
i_idxs_selected = [i_idxs[i] for i in ordered_indices[:n]]


# USCC-VQE solver
lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
mc_uscc.fcisolver.norb_f = ncas_f
mc_uscc.kernel(ci0=las_ci0_f)
print("LASUSCCSD-VQE energy: {:.9f}".format(mc_uscc.e_tot))


# traditional LSCC solver
lscc_fci = FCISolver_CC(mol, a_idxs_selected, i_idxs_selected)
mc_uscc.fcisolver = lscc_fci
mc_uscc.fcisolver.norb_f = ncas_f
mc_uscc.kernel(ci0=las_ci0_f)
print("LASUSCCSD-CC energy: {:.9f}".format(mc_uscc.e_tot))

print("S in the Ui basis\n")
print_list_matrix(lscc_fci.S)
print("H in the Ui basis\n")
print_list_matrix(lscc_fci.H)
print("lccsi Ui basis\n")
print_list_matrix(lscc_fci.lccsi)


# reproducing the same H and S through the Op class
h1eff,e_core= mc_uscc.get_h1eff(mc_uscc.mo_coeff)
h2eff = mc_uscc.get_h2eff() 
h = [e_core, h1eff, h2eff]
fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
psi = lasci_ominus1.LASUCCTrialState(fcisolver, las_ci0_f, norb = 4, norb_f = [2,2], nelec=[2,2])
c, uc, huc = psi.hc_x (np.zeros_like(psi.x), h)[0:3]


nmo = las.mo_coeff.shape[1]
ncas, ncore = las.ncas, las.ncore
nocc = ncore + ncas
h2e =  lib.numpy_helper.unpack_tril (las.get_h2eff().reshape (nmo*ncas,ncas*(ncas+1)//2)).reshape (nmo, ncas, ncas, ncas)[ncore:nocc,:,:,:]
h1las, h0las = las.h1e_for_cas(mo_coeff=las.mo_coeff)
h2las = h2e

h1lasspin = grad.get_h1e_spin(h1las)
h2lasspin = grad.get_eri_spin(h2las).transpose(3,0,2,1)

from helper.op import Op, IdentityOp, UOp, h1Op, h2Op
Hop = h1Op(h1lasspin) + 0.5 * h2Op(h2lasspin) + e_core * IdentityOp()

# Now we reproduce the same H and S through the Op class
Uops = [IdentityOp()] + [UOp(np.pi / 2,a_idxs_selected[i], i_idxs_selected[i]) for i in range(len(a_idxs_selected))]

UH = np.zeros((len(Uops), len(Uops)), dtype=np.complex128)
US = np.zeros((len(Uops), len(Uops)), dtype=np.complex128)
for i, op in enumerate(Uops):
    for j, op2 in enumerate(Uops):
        US[i,j] = np.vdot(c.ravel(), op.dagger().apply(op2.apply(c.ravel())))
        UH[i,j] = np.vdot(c.ravel(), op.dagger().apply(Hop.apply(op2.apply(c.ravel()))))

print("S in the Ui basis (from Op class)\n")
print_list_matrix(US)
print("H in the Ui basis (from Op class)\n")
print_list_matrix(UH)

assert np.allclose(US, lscc_fci.S), "Overlap matrices should match"
assert np.allclose(UH, lscc_fci.H), "Hamiltonian matrices should match"

# now we diagonalize in the Ai basis
Aops = [IdentityOp()]
for a_idx, i_idx in zip(a_idxs_selected, i_idxs_selected):
    Aterms = [
        (1, [("annihilate",i ) for i in i_idx] + [("create",a) for a in a_idx[::-1]]),
        (-1,[("annihilate",a) for a in a_idx] + [("create", i) for i in i_idx[::-1]]),
    ]
    Aops.append(Op(Aterms))

AH = np.zeros((len(Aops), len(Aops)), dtype=np.complex128)
AS = np.zeros((len(Aops), len(Aops)), dtype=np.complex128)
for i, op in enumerate(Aops):
    for j, op2 in enumerate(Aops):
        AS[i,j] = np.vdot(c.ravel(), op.dagger().apply(op2.apply(c.ravel())))
        AH[i,j] = np.vdot(c.ravel(), op.dagger().apply(Hop.apply(op2.apply(c.ravel()))))

print("S in the Ai basis (from Op class)\n")
print_list_matrix(AS)
print("H in the Ai basis (from Op class)\n")
print_list_matrix(AH)

# Diagonalize the generalized eigenvalue problem AH c = E AS c
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(AS) @ AH)
idx = np.argsort(eigenvalues)
ground_state_energy = eigenvalues[idx[0]]
ground_state_coeffs = eigenvectors[:, idx[0]]

print("Ground state energy: {:.9f}".format(ground_state_energy))
print("Ground state coefficients:\n", ground_state_coeffs)


# A1terms = [
#     (1, [("annihilate", i) for i in i_idxs[0]] + [("create", a) for a in a_idxs[0][::-1]]),
#     (-1,[("annihilate", a) for a in a_idxs[0]] + [("create", i) for i in i_idxs[0][::-1]]),
# ]
# A1 = Op(A1terms)
# excitation_ops = [IdentityOp(), A1]

# H = np.zeros((len(excitation_ops), len(excitation_ops)), dtype=np.complex128)
# S = np.zeros((len(excitation_ops), len(excitation_ops)), dtype=np.complex128)
# for i, op in enumerate(excitation_ops):
#     for j, op2 in enumerate(excitation_ops):
#         S[i,j] = np.vdot(c.ravel(), op.dagger().apply(op2.apply(c.ravel())))
#         H[i,j] = np.vdot(c.ravel(), op.dagger().apply(Hop.apply(op2.apply(c.ravel()))))

# # Diagonalize the generalized eigenvalue problem HSc = ESc
# eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S) @ H)
# print("Eigenvalues (energies):\n", eigenvalues)
# print("Eigenvectors:\n", eigenvectors)
