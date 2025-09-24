import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi
from mrh.my_pyscf.tools import molden
from c2h4n4_struct import structure as struct

from mrh.exploratory.citools import grad, lasci_ominus1, fockspace
from mrh.exploratory.unitary_cc import lasuccsd
# Using LASSI[r,q]
from scipy.linalg import eigh

from mrh.my_pyscf.lassi import op_o0
from mrh.my_pyscf.lassi import op_o1


from pyscf.fci import cistring
from pyscf import fci
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci.spin_op import contract_ss

def cilas2f(lasci, norb_f, nelec_f):
    ''' nelec (na, nb)'''
    ci_f = []
    for i, ci in enumerate(lasci):
        ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
    return ci_f


def addr_outer_product (norb_f, nelec_f):
    norb = sum (norb_f)
    nelec = sum (nelec_f)
    # Must skip over cases where there are no electrons of a specific spin in a particular subspace
    norbrange = np.cumsum (norb_f)
    addrs = []
    for i in range (0, len (norbrange)):
        new_addrs = cistring.sub_addrs (norb, nelec, range (norbrange[i]-norb_f[i], norbrange[i]), nelec_f[i]) if nelec_f[i] else []
        if len (addrs) == 0:
            addrs = new_addrs
        elif len (new_addrs) > 0:
            addrs = np.intersect1d (addrs, new_addrs)
    return addrs

def _ci_outer_product (ci_f, norb_f, nelec_f):
    # There may be an ambiguous factor of -1, but it should apply to the entire product CI vector so maybe it doesn't matter?
    ci_dp = ci_f[-1].copy ()
    for ci_r in ci_f[-2::-1]:
        ndeta_1, ndetb_1 = ci_dp.shape
        ndeta_2, ndetb_2 = ci_r.shape
        ci_dp = np.multiply.outer (ci_dp, ci_r)
        ci_dp = ci_dp.transpose (0,2,1,3).reshape (ndeta_1*ndeta_2, ndetb_1*ndetb_2)
    neleca_f = [ne[0] for ne in nelec_f]
    nelecb_f = [ne[1] for ne in nelec_f]
    addrs_a = addr_outer_product (norb_f, neleca_f)
    addrs_b = addr_outer_product (norb_f, nelecb_f)
    ci = np.zeros ((cistring.num_strings (sum (norb_f), sum (neleca_f)), cistring.num_strings (sum (norb_f), sum (nelecb_f))),
        dtype=ci_dp.dtype)
    ci[np.ix_(addrs_a,addrs_b)] = ci_dp[:,:] / linalg.norm (ci_dp)
    return ci

def ci_outer_product (ci_fr, norb_f, nelec_fr):
    ci_r = []
    for state in range (len (ci_fr[0])):
        ci_f = [ci[state] for ci in ci_fr]
        nelec_f = [nelec[state] for nelec in nelec_fr]
        ci_r.append (_ci_outer_product (ci_f, norb_f, nelec_f))
    nelec = (sum ([ne[0] for ne in nelec_f]),
             sum ([ne[1] for ne in nelec_f]))
    return ci_r, nelec

def slow_ham (mol, h1, h2, ci_fr, norb_f, nelec_fr, orbsym=None):
    ci, nelec = ci_outer_product (ci_fr, norb_f, nelec_fr)
    solver = fci.solver (mol).set (orbsym=orbsym)
    norb = sum (norb_f)
    h2eff = solver.absorb_h1e (h1, h2, norb, nelec, 0.5)
    ham_ci = [solver.contract_2e (h2eff, c, norb, nelec) for c in ci]
    s2_ci = [contract_ss (c, norb, nelec) for c in ci]
    ham_eff = np.array ([[c.ravel ().dot (hc.ravel ()) for hc in ham_ci] for c in ci])
    s2_eff = np.array ([[c.ravel ().dot (s2c.ravel ()) for s2c in s2_ci] for c in ci])
    ovlp_eff = np.array ([[bra.ravel ().dot (ket.ravel ()) for ket in ci] for bra in ci])
    return ham_eff, s2_eff, ovlp_eff

def compute_xhx(ham, ovlp, thresh=1e-12):
    # Ensure Hermiticity
    ham = (ham + ham.conj().T) / 2
    ovlp = (ovlp + ovlp.conj().T) / 2
    
    # Eigen-decomposition of overlap matrix
    s_vals, s_vecs = np.linalg.eigh(ovlp)
    
    # Discard numerically tiny eigenvalues (linear dependencies)
    keep = s_vals > thresh
    if not np.all(keep):
        s_vals = s_vals[keep]
        s_vecs = s_vecs[:, keep]
    
    # Construct S^{-1/2}
    s_inv_sqrt = np.diag(1.0 / np.sqrt(s_vals))
    X = s_vecs @ s_inv_sqrt @ s_vecs.conj().T
    
    # Transform Hamiltonian
    xhx = X.conj().T @ ham @ X
    
    return xhx, X

def get_Sij_Hij(psi_i, psi_j, h):
    ucj, hucj = psi_j.hc_x (psi_j.x, h)[1:3]
    uci, huci = psi_i.hc_x (psi_i.x, h)[1:3]
    ucj, hucj = ucj.ravel (), hucj.ravel ()
    uci = uci.ravel ()
    Sij = uci.conj ().dot (ucj)
    Hij = uci.conj ().dot (hucj)
    return Sij, Hij

def cilasf(lasci, norb_f, nelec_f):
    ''' nelec (na, nb)'''
    ci_f = []
    for i, ci in enumerate(lasci):
        ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
    return ci_f

def split_grouped_ci(ci):
    if len(ci) == 1 and isinstance(ci[0], list):
        return [[arr] for arr in ci[0]]
    else:
        # Already in target format, or not grouped
        return ci

def to_nested_format(ci_list):
    """
    Transform a list of numpy arrays into a nested list format,
    where each array is wrapped inside its own list.

    Example:
        [A, B]  -->  [[A], [B]]
    """
    return [[split_grouped_ci(arr)] for arr in ci_list]


def print_matrix(mat):
    for row in mat:
        print("  ".join(f"{x:.17f}" for x in row))

mol = struct (0, 0, '6-31g')
mol.output = 'c2h4n4_lassi_631g.log'
mol.verbose = lib.logger.INFO
mol.build ()
mf = scf.RHF (mol).run ()

las = LASSCF (mf, (3,3), ((2,1),(1,2)))

# For this molecule we use 2 reference states
# and create 14 CT LAS states (7 for each reference)
# but in most molecules you may only need 1 reference state
las = las.state_average ([0.5,0.5],
    spins=[[1,-1],[-1,1]],
    smults=[[2,2],[2,2]],    
    charges=[[0,0],[0,0]])


# reorders the molecular-orbital (MO) coefficient matrix so that
# the orbitals you specify become the active-space (CAS) block.
init_mo = las.sort_mo ([16,18,22,23,24,26])
init_mo = las.localize_init_guess ((list (range (5)), list (range (5,10))), init_mo)
h2eff_sub, veff = las.kernel (init_mo)[-2:]


# lower bound method to compare 
mc = mcscf.CASCI (mf, 6, 6).set (fcisolver=csf_solver(mol,smult=1))
mc.kernel (las.mo_coeff)

print ("LASSCF energy =", las.e_tot)
print ("CASCI energy =", mc.e_tot)


# LASSIrq generates CT states using LAS reference states
r = 1
q = 1
lsi = lassi.LASSIrq(las,r=r,q=q)
e_roots, si_rq = lsi.kernel()


# prepare all the necessary objects for state interaction
ncore, ncas, nocc = las.ncore, las.ncas, las.ncore + las.ncas
nelecas = sum(las.nelecas_sub)
mo_coeff = las.mo_coeff
mo_core = mo_coeff[:,:ncore]
mo_cas = mo_coeff[:,ncore:nocc]
e0 = las._scf.energy_nuc () + 2 * (((las._scf.get_hcore () + veff.c/2) @ mo_core) * mo_core).sum () 
h1 = mo_cas.conj ().T @ (las._scf.get_hcore () + veff.c) @ mo_cas
h2 = h2eff_sub[ncore:nocc].reshape (ncas*ncas, ncas * (ncas+1) // 2)
h2 = lib.numpy_helper.unpack_tril (h2).reshape (ncas, ncas, ncas, ncas)
nelec_fr = lsi.get_nelec_frs()


# compute the overlap matrix and Hamiltonian matrix in the CT las state basis
ham_eff, s2_eff, ovlp_eff = slow_ham (las.mol, h1, h2, lsi.ci, las.ncas_sub, nelec_fr)
print("ham_eff\n", ham_eff)
print("\novlap_eff\n", ovlp_eff)

# Diagonalize the generalized eigenvalue problem
xhx = compute_xhx(ham_eff, ovlp_eff)[0]
e, c = linalg.eigh (xhx)
print("Eigenvalues\n", e)


# using vqe to optimize each CT states
# nstates = len(lsi.ci[0])
nstates = 5
psis = []

for i in range(nstates):
    ci = [[lsi.ci[fragj][i]] for fragj in range(len(lsi.ci))]
    nelec_sub = [nelec_fr[fragj][i] for fragj in range(len(nelec_fr))]
    
    tmplas = LASSCF (mf, (3,3), nelec_sub)
    tmplas.mo_coeff = mo_coeff
    tmplas.ci = ci
    all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(tmplas, 0.00001)

    nelecas_sub = [sum(nelec) for nelec in nelec_sub]
    mc_uscc = mcscf.CASCI(mf, ncas, nelecas)
    mc_uscc.mo_coeff = mo_coeff
    mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol,a_idxs_selected,i_idxs_selected)
    mc_uscc.fcisolver.norb_f = las.ncas_sub
    fci = mc_uscc.fcisolver
    lasci_ominus1.GLOBAL_MAX_CYCLE = 10
    mc_uscc.kernel(ci0 = cilas2f(ci, las.ncas_sub, nelec_sub))
    psis.append(mc_uscc.fcisolver.psi)

h1eff,e_core= mc_uscc.get_h1eff(mc_uscc.mo_coeff)
h2eff = mc_uscc.get_h2eff()
h = [e_core, h1eff, h2eff]

print("h1eff = \n", h1eff)
print("h2eff = \n", h2eff) 

nc = len(psis)

S = np.zeros((nc, nc), dtype=np.complex128)
H = np.zeros((nc, nc), dtype=np.complex128)

for i in range(nc):
    for j in range(nc):
        S[i, j], H[i, j] = get_Sij_Hij(psis[i], psis[j], h)

print("S matrix \n")
print_matrix(S)
print("H matrix \n")
print_matrix(H)

eigvals, eigvecs = eigh(H, S)
print("lowest eigenvalue = ", eigvals[0])
print("lowest eigenvector = \n", eigvecs[:, 0])


# compare the energy without VQE
lsi = lassi.LASSI(las)
e_roots, si_hand = lsi.kernel()
print ("LASSI(hand) energy =", e_roots)
print ("SI vector (hand):")
print (si_hand[:,0])



