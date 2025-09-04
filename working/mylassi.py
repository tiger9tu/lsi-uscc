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

def cilas2f(lasci, norb_f, nelec_f):
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
las = las.state_average ([0.5,0.5],
    spins=[[1,-1],[-1,1]],
    smults=[[2,2],[2,2]],    
    charges=[[0,0],[0,0]])
mo = las.sort_mo ([16,18,22,23,24,26])
mo = las.localize_init_guess ((list (range (5)), list (range (5,10))), mo)
las.kernel (mo)
molden.from_lasscf (las, 'c2h4n4_lasscf66_631g.molden')

mc = mcscf.CASCI (mf, 6, 6).set (fcisolver=csf_solver(mol,smult=1))
mc.kernel (las.mo_coeff)
molden.from_mcscf (mc, 'c2h4n4_casscf66_631g.molden', cas_natorb=True)

print ("LASSCF((3,3),(3,3)) energy =", las.e_tot)
print ("CASCI(6,6) energy =", mc.e_tot)

las2 = las.state_average ([0.5,0.5,0,0],
    spins=[[1,-1],[-1,1],[0,0],[0,0]],
    smults=[[2,2],[2,2],[1,1],[1,1]],    
    charges=[[0,0],[0,0],[-1,1],[1,-1]])
las2.lasci ()
las2.dump_spaces ()


h2eff_sub, veff = las2.kernel (mo)[-2:]
e_states = las2.e_states

ncore, ncas, nocc = las2.ncore, las2.ncas, las2.ncore + las2.ncas
mo_coeff = las2.mo_coeff
mo_core = mo_coeff[:,:ncore]
mo_cas = mo_coeff[:,ncore:nocc]
e0 = las2._scf.energy_nuc () + 2 * (((las2._scf.get_hcore () + veff.c/2) @ mo_core) * mo_core).sum () 
h1 = mo_cas.conj ().T @ (las2._scf.get_hcore () + veff.c) @ mo_cas
h2 = h2eff_sub[ncore:nocc].reshape (ncas*ncas, ncas * (ncas+1) // 2)
h2 = lib.numpy_helper.unpack_tril (h2).reshape (ncas, ncas, ncas, ncas)
nelec_fr = []
for fcibox, nelec in zip (las2.fciboxes, las2.nelecas_sub):
    ne = sum (nelec)
    nelec_fr.append ([_unpack_nelec (fcibox._get_nelec (solver, ne)) for solver in fcibox.fcisolvers])

print("nelec_fr = ", nelec_fr)
ham_eff, s2_eff, ovlp_eff = slow_ham (las2.mol, h1, h2, las2.ci, las2.ncas_sub, nelec_fr)
print("ham_eff\n", ham_eff)
print("\novlap_eff\n", ovlp_eff)
print (las.converged, e_states - (e0 + np.diag (ham_eff)))
xhx = compute_xhx(ham_eff, ovlp_eff)[0]
print("X^dag H X\n", xhx)
e, c = linalg.eigh (xhx)

print("Eigenvalues\n", e)


nstates = 4
# cis, nelecs = ci_outer_product (las2.ci, las2.ncas_sub, nelec_fr)
print("las2.ncas_sub = ", las2.ncas_sub)
print("nelecfr = ", nelec_fr)
psis = []
ncas = 6
nelecas = 6
for i in range(nstates):
    ci = [[las2.ci[fragj][i]] for fragj in range(len(las2.ci))]
    # ci = cis[i]
    # print("ci = \n", ci)
    ncas_sub = las2.ncas_sub
    nelec_sub = [nelec_fr[fragj][i] for fragj in range(len(nelec_fr))]
    print("nelec_sub = ", nelec_sub)
    nelecas_sub = [sum(nelec) for nelec in nelec_sub]
    print("nelecas = ", nelecas_sub)
    mc_uscc = mcscf.CASCI(mf, ncas, nelecas)
    mc_uscc.mo_coeff = las2.mo_coeff
    mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol,[],[])
    mc_uscc.fcisolver.norb_f = ncas_sub
    fci = mc_uscc.fcisolver
    psis.append(fci.build_psi(cilas2f(ci, ncas_sub, nelec_sub), ncas, ncas_sub, nelecas_sub))

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



# # all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(las2, 0.001)
# ncas = 6
# nelecas = 6

# mc_uscc = mcscf.CASCI(mf, ncas, nelecas)
# mc_uscc.mo_coeff = las2.mo_coeff
# mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, [], [])
# mc_uscc.fcisolver.norb_f = [3,3]
# fci = mc_uscc.fcisolver
# # # easily hit the maximal memory limit
# # # mc_uscc.fcisolver.frozen = test_config['frozen'] if 'frozen' in test_config else None  
# # # mc_uscc.kernel(ci0 = cilas2f(las2.ci, mol_config['ncas'], mol_config['nelecas']))

# # # first verify that we can reproduce the energy of LASSI
# psis = []

# # print("nested[0]: \n", to_nested_format(las2.ci)[0])

# for i in range(len(las2.ci[0])):
#     cii = []
#     for j in range(len(las2.ci)):
#         cii.append(las2.ci[j][i])
#     psis.append(fci.build_psi(cilas2f(cii, las2.ncas_sub, las2.nelecas_sub), ncas, las2.ncas_sub, nelecas))

# h1eff,e_core= mc_uscc.get_h1eff(mc_uscc.mo_coeff)
# h2eff = mc_uscc.get_h2eff()
# h = [e_core, h1eff, h2eff]

# print("h1eff = \n", h1eff)
# print("h2eff = \n", h2eff) 

# nc = len(psis)

# S = np.zeros((nc, nc), dtype=np.complex128)
# H = np.zeros((nc, nc), dtype=np.complex128)

# for i in range(nc):
#     for j in range(nc):
#         S[i, j], H[i, j] = get_Sij_Hij(psis[i], psis[j], h)

# print("S matrix \n")
# print_matrix(S)
# print("H matrix \n")
# print_matrix(H)

# # print("las e_tot = ", las_e_tot)

# eigvals, eigvecs = eigh(H, S)
# print("lowest eigenvalue = ", eigvals[0])
# print("lowest eigenvector = \n", eigvecs[:, 0])

lsi = lassi.LASSI(las2)
e_roots, si_hand = lsi.kernel()
print ("LASSI(hand) energy =", e_roots)
print ("SI vector (hand):")
print (si_hand[:,0])


# molden.from_lassi (lsi, 'c2h4n4_lassi_631g.molden', state=0)

