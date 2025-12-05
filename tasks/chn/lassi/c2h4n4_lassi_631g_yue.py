import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi
from mrh.my_pyscf.tools import molden
from mrh.exploratory.citools import grad, lasci_ominus1
from mrh.exploratory.unitary_cc import lasuccsd
from c2h4n4_struct import structure as struct
from helper import util
# Using a hand-made model space

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

# las2 = las.state_average ([0.5,0.5,0,0],
#     spins=[[1,-1],[-1,1],[0,0],[0,0]],
#     smults=[[2,2],[2,2],[1,1],[1,1]],    
#     charges=[[0,0],[0,0],[-1,1],[1,-1]])
las2 = las
las2.lasci ()
las2.dump_spaces ()
print("las2 e_tot = ", las2.e_tot)


lsi = lassi.LASSI(las2)
e_roots, si_hand = lsi.kernel()
print ("LASSI(hand) energy =", e_roots[0])
print ("SI vector (hand):")
print (si_hand[:,0])



# let's try diagonalize them in fci basis
# las0ci = np.array(las2.ci)[:,0,:,:]
# las1ci = np.array(las2.ci)[:,1,:,:]
# las2ci = np.array(las2.ci)[:,2,:,:]
# las3ci = np.array(las2.ci)[:,3,:,:]

# las0ci_f =  util.cilas2f(las0ci, [3,3], ((2,1),(1,2)))
# las1ci_f =  util.cilas2f(las1ci, [3,3], ((2,1),(1,2)))
# las2ci_f =  util.cilas2f(las2ci, [3,3], ((2,1),(1,2)))
# las3ci_f =  util.cilas2f(las3ci, [3,3], ((2,1),(1,2)))

# for lasci_f in [las0ci_f, las1ci_f, las2ci_f, las3ci_f]:
#     for ci in lasci_f:
#         ci /= linalg.norm(ci)

# mc_uscc =  mcscf.CASCI (mf, 6, 6)
# mc_uscc.mo_coeff = las2.mo_coeff
# lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
# mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, [], [])
# mc_uscc.fcisolver.norb_f = [3,3]
# fci = mc_uscc.fcisolver

# h1eff, e_core = mc_uscc.get_h1eff (mc_uscc.mo_coeff)
# h2eff = mc_uscc.get_h2eff (mc_uscc.mo_coeff)
# h = [e_core, h1eff, h2eff]

# psi0 = getattr (fci, 'psi', fci.build_psi (las0ci_f, 6, (3,3), 6))
# psi1 = getattr (fci, 'psi', fci.build_psi (las1ci_f, 6, (3,3), 6))
# psi2 = getattr (fci, 'psi', fci.build_psi (las2ci_f, 6, (3,3), 6))
# psi3 = getattr (fci, 'psi', fci.build_psi (las3ci_f, 6, (3,3), 6))


# def get_Sij_Hij(psi_i, psi_j, h):
#     ucj, hucj = psi_j.hc_x (psi_j.x, h)[1:3]
#     uci, huci = psi_i.hc_x (psi_i.x, h)[1:3]
#     ucj, hucj = ucj.ravel (), hucj.ravel ()
#     uci = uci.ravel ()
#     Sij = uci.conj ().dot (ucj)
#     Hij = uci.conj ().dot (hucj)
#     return Sij, Hij

# psis = [psi0, psi1, psi2, psi3]
las0ci = np.array(las2.ci)[:,0,:,:]
las1ci = np.array(las2.ci)[:,1,:,:]
spins=[[1,-1],[-1,1]]
# las2ci = np.array(las2.ci)[:,2,:,:]
# las3ci = np.array(las2.ci)[:,3,:,:]
nfrag = 2
charges = 0 # later should be replaced to charges[i][j]
neleca = [(sum(las2.nelecas_sub[j]) - charges + spins[i][j]) // 2 for j in range(nfrag) for i in range(2)]
nelecb = [(sum(las2.nelecas_sub[j]) - charges - spins[i][j]) // 2 for j in range(nfrag) for i in range(2)]
print("neleca = ", neleca)
print("nelecb = ", nelecb)


las0ci_f =  util.cilas2f(las0ci, [3,3], ((2,1),(1,2)))
las1ci_f =  util.cilas2f(las1ci, [3,3], ((1,2),(2,1)))
# las2ci_f =  util.cilas2f(las2ci, [3,3], ((2,1),(1,2)))
# las3ci_f =  util.cilas2f(las3ci, [3,3], ((2,1),(1,2)))

for lasci_f in [las0ci_f, las1ci_f]:
    for ci in lasci_f:
        ci /= linalg.norm(ci)

mc_uscc =  mcscf.CASCI (mf, 6, 6)
mc_uscc.mo_coeff = las2.mo_coeff
lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, [], [])
mc_uscc.fcisolver.norb_f = [3,3]
fci = mc_uscc.fcisolver

h1eff, e_core = mc_uscc.get_h1eff (mc_uscc.mo_coeff)
h2eff = mc_uscc.get_h2eff (mc_uscc.mo_coeff)
h = [e_core, h1eff, h2eff]

psi0 = getattr (fci, 'psi', fci.build_psi (las0ci_f, 6, (3,3), 6))
psi1 = getattr (fci, 'psi', fci.build_psi (las1ci_f, 6, (3,3), 6))
# psi2 = getattr (fci, 'psi', fci.build_psi (las2ci_f, 6, (3,3), 6))
# psi3 = getattr (fci, 'psi', fci.build_psi (las3ci_f, 6, (3,3), 6))


def get_Sij_Hij(psi_i, psi_j, h):
    ucj, hucj = psi_j.hc_x (psi_j.x, h)[1:3]
    uci, huci = psi_i.hc_x (psi_i.x, h)[1:3]
    ucj, hucj = ucj.ravel (), hucj.ravel ()
    uci = uci.ravel ()
    Sij = uci.conj ().dot (ucj)
    Hij = uci.conj ().dot (hucj)
    return Sij, Hij

psis = [psi0, psi1]


n = len (psis)
S = np.zeros ((n,n), dtype=complex)
H = np.zeros ((n,n), dtype=complex)

for i in range(n):
    for j in range(n):
        S[i,j], H[i,j] = get_Sij_Hij (psis[i], psis[j], h)

print("Overlap matrix S:")
util.print_list_matrix (S, digits=4)
print("Hamiltonian matrix H:")
util.print_list_matrix (H, digits=4)

e_vals, e_vecs = linalg.eig (H, S)
idx = e_vals.argsort ()
e_vals = e_vals[idx]
e_vecs = e_vecs[:,idx]
print("Diagonalized LASSI energies from FCI basis:")
for i in range(n):
    print("State {:.0f} energy = {:.9f}".format(i, e_vals[i].real))
    print("  Coefficients: ", e_vecs[:,i].real)


# molden.from_lassi (lsi, 'c2h4n4_lassi_631g.molden', state=0)

# c = si_hand[:,0]
# lsi_ground_ci = (las2.ci * c[None,:,None,None]).sum(axis=1)
# print("LASIS all ci")
# util.print_list_matrix(las2.ci)
# print("LASSI gound ci")
# util.print_list_matrix(lsi_ground_ci)


# lsi_ground_ci_f =  util.cilas2f(lsi_ground_ci, [3,3], ((2,1),(1,2)))

# for ci in lsi_ground_ci_f:
#     ci /= linalg.norm(ci)

# lsirdm1s, lsirdm2s = lsi.make_casdm12s ()
# lsi_ground_rdm1s = lsirdm1s[0]
# lsi_ground_rdm2s = util.lassi_rdm2_to_lasscf (lsirdm2s[0])
# g_sel, grad_all, a_idxs_selected, i_idxs_selected = grad.get_grad_exact_rdm12 (las2, lsi_ground_rdm1s, lsi_ground_rdm2s, epsilon=0.001)
# a_idxs, i_idxs = util.get_sorted_excitations (a_idxs_selected, i_idxs_selected, g_sel, fraction = 0.01, verbose=3)

# mc_uscc =  mcscf.CASCI (mf, 6, 6)
# mc_uscc.mo_coeff = las2.mo_coeff
# lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
# mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs, i_idxs)
# mc_uscc.fcisolver.norb_f = [3,3]
# fci = mc_uscc.fcisolver

# h1eff, e_core = mc_uscc.get_h1eff (mc_uscc.mo_coeff)
# h2eff = mc_uscc.get_h2eff (mc_uscc.mo_coeff)
# h = [e_core, h1eff, h2eff]
# print("lsi_ground_ci_f")
# util.print_list_matrix(lsi_ground_ci_f)

# psi = getattr (fci, 'psi', fci.build_psi (lsi_ground_ci_f, 6, (3,3), 6))

# fci = psi.dp_ci(psi.ci_f)
# print("fci norm = ", linalg.norm(fci))

# elasipsi = psi.energy_tot(psi.x, h)
# print("Initial energy from LAS-CI0: {:.9f}".format(elasipsi))