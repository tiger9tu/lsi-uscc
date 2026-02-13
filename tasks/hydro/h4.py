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

# fracs = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08]
fracs = [0.01]
mc_uscc = mcscf.CASCI(mf, sum(ncas_f), sum(nelecas_f))
mc_uscc.mo_coeff = las.mo_coeff

lscc_fci = None

# for frac in fracs:
frac = fracs[0]
n = max(1, int(n_excitations * frac))
a_idxs_selected = [a_idxs[i] for i in ordered_indices[:n]]
i_idxs_selected = [i_idxs[i] for i in ordered_indices[:n]]
norb = 4
for a, i in zip (a_idxs_selected, i_idxs_selected):
    print ("a, i = ", a, i)
    errstr = 'a,i={},{} breaks sz symmetry'.format (a, i)
    #print ("SV sum orb = ",np.sum(a // norb))
    #print ("SV errstr = ", errstr)
    assert (np.sum (a//norb) == np.sum (i//norb)), errstr

# las_rdm2 = las.make_r
print(f"\nFraction: {frac} | Number of excitations: {n}")


#Computing energy through the LAS-UCC kernel using selected excitations
#==========================================================================================

lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
mc_uscc.fcisolver.norb_f = ncas_f
mc_uscc.kernel(ci0=las_ci0_f)
print("LASUSCCSD-VQE energy: {:.9f}".format(mc_uscc.e_tot))
# a_idxs_selected.insert(0, np.array([], dtype=np.uint8))
# i_idxs_selected.insert(0, np.array([], dtype=np.uint8))



print("Selected excitations (a_idxs, i_idxs):")
for a_idx, i_idx in zip(a_idxs_selected, i_idxs_selected):
    print(f"  a_idx: {a_idx}, i_idx: {i_idx}")
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
print("LASLSCCSD energy: {:.9f}".format(mc_uscc.e_tot))
S = mc_uscc.fcisolver.S
H = mc_uscc.fcisolver.H
print("S matrix:\n")
print_list_matrix(S)
print("H matrix:\n")
print_list_matrix(H)


from helper.op import Op, IdentityOp, UOp, h1Op, h2Op





def print_direc(fcivec, threshold=1e-6):
    nbits = max(1, (fcivec.size - 1).bit_length())
    for idx, amp in enumerate(fcivec):
        if abs(amp) > threshold:
            det = format(idx, f"0{nbits}b")
            print(f"{amp} |{det}>")

# First we test the fci case for simplicity
psi = mc_uscc.fcisolver.las_psi0s[0]
fcivec = psi.dp_ci(psi.ci_f)
# fcivec_vec = fcivec.transpose().reshape(fcivec.size)
print("FCI CI vector (sparse format):")
fcivec_vec = fcivec.transpose().reshape(fcivec.size)
print_direc(fcivec_vec)


ucc_ops = [IdentityOp()] + [UOp(np.pi/2, a_idx, i_idx) for a_idx, i_idx in zip(a_idxs_selected, i_idxs_selected)]

Ufcivec = ucc_ops[1].apply(fcivec_vec)
print("After applying the first UOp (pi/2 rotation):")
print_direc(Ufcivec)
for i, op1 in enumerate(ucc_ops):
    for j, op2 in enumerate(ucc_ops):
        op = op1.conjugate() * op2
        new_vec = op.apply(fcivec_vec)
        overlap = np.dot(fcivec_vec.conj(), new_vec)
        print(f"Overlap Sij = {overlap}")

# Generate OEI, TEI
nmo = las.mo_coeff.shape[1]
ncas, ncore = las.ncas, las.ncore
nocc = ncore + ncas
h2e = lib.numpy_helper.unpack_tril (las.get_h2eff().reshape (nmo*ncas,ncas*(ncas+1)//2)).reshape (nmo, ncas, ncas, ncas)[ncore:nocc,:,:,:]
h1las, h0las = las.h1e_for_cas(mo_coeff=las.mo_coeff)
h2las = h2e


# now let's try evaluate <LAS | H | LAS >
h = h1_op + h2_op
H_LAS = h.apply(fcivec_vec)
energy = np.dot(fcivec_vec.conj(), H_LAS)
print(f"Energy from direct application of H operator: {energy}")



h1t1 = grad.get_grad_h1t1([[2]], [[3]], las.make_casdm1s(), h1las)
print(f"Gradient h1t1 sum: {np.sum(h1t1)}")


#
# for evaluate the S matrix, we iterate through all the pairs of excitations
# for p, (a_idx_p, i_idx_p) in enumerate(zip(a_idxs_selected, i_idxs_selected)):
#     for q, (a_idx_q, i_idx_q) in enumerate(zip(a_idxs_selected, i_idxs_selected)):
#         if a_idx_p.size == 0 and i_idx_p.size == 0:
#             Up_pi_2 = IdentityOp()
#         else:
#             Up_pi_2 = UOp(np.pi/2, a_idx_p, i_idx_p)
#         if a_idx_q.size == 0 and i_idx_q.size == 0:
#             Uq_pi_2 = IdentityOp()
#         else:
#             Uq_pi_2 = UOp(np.pi/2, a_idx_q, i_idx_q)

#         Up_diag_Uq = Up_pi_2.conjugate() * Uq_pi_2


        # <Up(pi/2)' Uq(pi/2)> = <(1 + ap'ip - ip'ap - (nap + nip - 2 nap nip))' (1 + aq'iq - iq'aq - (naq + niq - 2 naq niq))>
        
        
        # we have to get all the combinations of the above terms
        # but for now let's do it for our simple test case, which is 

        # <Ap* Aq> = <(ap'ip)' aq'iq> = <ip' ap aq' iq>  
        # first, we obtain the spatial orbital and spin for each index in a_idx and i_idx
        # a_idx_p_spatial = [idx % norb for idx in a_idx_p]
        # a_idx_p_spin = [idx // norb for idx in a_idx_p]
        # i_idx_p_spatial = [idx % norb for idx in i_idx_p]
        # i_idx_p_spin = [idx // norb for idx in i_idx_p]

        # a_idx_q_spatial = [idx % norb for idx in a_idx_q]
        # a_idx_q_spin = [idx // norb for idx in a_idx_q]
        # i_idx_q_spatial = [idx % norb for idx in i_idx_q]
        # i_idx_q_spin = [idx // norb for idx in i_idx_q]

        # ops_spatial = i_idx_p_spatial + a_idx_p_spatial + a_idx_q_spatial + i_idx_q_spatial
        # ops_spin = i_idx_p_spin + a_idx_p_spin + a_idx_q_spin + i_idx_q_spin
        # ops_type = [0] * len(i_idx_p) + [1] * len(a_idx_p) + [1] * len(a_idx_q) + [0] * len(i_idx_q)

        # frag_ops = [None] * n_frag

        # for op in range(len(ops_spatial)):
        #     spatial_idx = ops_spatial[op]
        #     spin_idx = ops_spin[op]
        #     op_type = ops_type[op]
        #     for frag_idx, frag in enumerate(frag_orbs):
        #         if spatial_idx in frag:
        #             if frag_ops[frag_idx] is None:
        #                 frag_ops[frag_idx] = ([], [])
        #             frag_spatial_idx = frag.index(spatial_idx)
        #             frag_ops[frag_idx][0].append(frag_spatial_idx + spin_idx * len(frag)) # convert to spin orbital index
        #             frag_ops[frag_idx][1].append(op_type)
        #             break

            
        
        # # we apply the excitation p followed by the de-excitation q to the reference determinant
        # orb_idxs = list(a_idx_p) + list(i_idx_p) + list(i_idx_q) + list(a_idx_q)
        # types = [1] * len(a_idx_p) + [0] * len(i_idx_p) + [1] * len(i_idx_q) + [0] * len(a_idx_q)

        # ci_new = op(orb_idxs, types, las_ci0_f)
        # S_pq = np.dot(las_ci0_f.conj(), ci_new)
        # print(f"S[{p}, {q}] = {S_pq}")