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



# # Now lets try the efficient  of fragmented evaluation
# def braAket(A, ci_f, norb_f):
#     # A is a single string of annihilation and creation operators defined on full space
#     # e.g. A = [("annihilate", 0), ("create", 4), ("annihilate", 1), ("create", 5)] = a_0^ c_4 a_1^ c_5

# Let's start first by only consider the a_idx: [5 1], i_idx: [6 0]
only_a_idx = [5, 1] 
only_i_idx = [6, 0]
# a0'a1' i1'i0'
# First perform jordan-wigner transformation implicitly to get the final per-site operators
nspin_orbs = 8

def jordan_wigner_res(ops, nspin_orbs):
    # apply jordan-wigner transformation implicitly to get the final per-site result
    # input: ops = (('annihilate', 7), ('annihilate', 5), ('create', 7), ('create', 7))
    # output: res_vaccum and res_occupied, are list of length nspin_orbs with abs values 1 (unoccupied), 2 (occupied), 3 (vanish), and signs for the phase
    # for example res_vaccum[3] = -2 means that when applying the jordan-wigner single site operators on |0>_3, we obtain -|1>_3
    res_vaccum = [1 for _ in range(nspin_orbs)] # 1 -> unoccupied, 2 -> occupied, +- sign, +-3 -> vanish
    res_occupied =[2 for _ in range(nspin_orbs)]
    for op_type, op_idx in ops:
        for res in [res_vaccum, res_occupied]:
            for k in range(op_idx):
                # s0,..,sk-1
                phase = -1 if abs(res[k]) == 2 else 1
                res[k] *= phase
            # bk
            if abs(res[op_idx]) == 2:
                if op_type == "annihilate":
                    res[op_idx] /= 2
                else:
                    res[op_idx] = 3
            elif abs(res[op_idx]) == 1:
                if op_type == "create":
                    res[op_idx] *= 2
                else:
                    res[op_idx] = 3
    
    return res_vaccum, res_occupied

def binarr(index, length):
    return [int(b) for b in format(index, f'0{length}b')]

def las_a_las_jwres(jwres_vaccum, jwres_occupied, ci_fs, frag_sorbs):
    # evaluate the <LAS| aiajak'am... |LAS>
    # input: jwres_vaccum and jwres_occupied are jw representations of operators
    # ci_f is the fragmented CI vector 
    inner_prod = 1
    for i, ci_f in enumerate(ci_fs):
        new_ci_f = np.zeros_like(ci_f.ravel())
        for det_idx, amp in enumerate(ci_f.ravel()):
            if abs(amp) < 1e-5:
                continue
            binary_array = binarr(det_idx , len(frag_sorbs[i]))
            res_bin_array = np.zeros_like(binary_array)
            coef = amp
            for forb_idx, occ in enumerate(binary_array[::-1]):
                orb_idx = frag_sorbs[i][forb_idx]
                if occ == 0:
                    res = jwres_vaccum[orb_idx]
                else:
                    res = jwres_occupied[orb_idx]
                if res == 3:
                    coef = 0
                    break
                if abs(res) == 1:
                    res_bin_array[forb_idx] = 0
                elif abs(res) == 2:
                    res_bin_array[forb_idx] = 1
                coef *= -1 if res < 0 else 1
            
            res_det_idx = int(''.join(map(str, res_bin_array[::-1])), 2)
            new_ci_f[res_det_idx] += coef
    
        inner_prod *= np.vdot(ci_f.ravel(), new_ci_f.ravel())
        if abs(inner_prod) < 1e-5:
            return 0
    return inner_prod


# First try evaluate <A' H A> for only only_a_idx = [5, 1]  only_i_idx = [6, 0]
frag_sorbs = [[0,1,4,5], [2,3,6,7]] # we have to generate frag_sorbs from frag_orbs
valueAHA = 0

tenpersent = len(Hop.terms) // 10
for i, term in enumerate(Hop.terms):
    # print("Evaluating term: ", terms)
    hcoef = term[0]
    jwres_vaccum, jwres_occupied = jordan_wigner_res(term[1], nspin_orbs)
    valueAHA += hcoef * las_a_las_jwres(jwres_vaccum, jwres_occupied, las_ci0_f, frag_sorbs)
    if (i+1) % tenpersent == 0:
        print(f"Progress: {(i+1) / len(Hop.terms) * 100:.1f}%")


print("Value of <A' H A> with only a_idx [5, 1] and i_idx [6, 0]: ", valueAHA)

