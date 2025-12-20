import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi
from mrh.my_pyscf.tools import molden
from c2h4n4_struct import structure as struct
from helper import util
from mrh.exploratory.citools import grad, lasci_ominus1
from mrh.exploratory.unitary_cc import lasuccsd
# Using LASSI[r,q]

def get_Sij_Hij(psi_i, psi_j, h):
    ucj, hucj = psi_j.hc_x (psi_j.x, h)[1:3]
    uci, huci = psi_i.hc_x (psi_i.x, h)[1:3]
    ucj, hucj = ucj.ravel (), hucj.ravel ()
    uci = uci.ravel ()
    Sij = uci.conj ().dot (ucj)
    Hij = uci.conj ().dot (hucj)
    return Sij, Hij


r = 2
q = 2

mol = struct (0, 0, '6-31g')
mol.output = 'c2h4n4_lassirq_631g.log'
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

lsi = lassi.LASSIrq(las,r=r,q=q)
e_roots, si_rq = lsi.kernel()


print ("LASSI[{},{}]energy =".format (r,q), e_roots[0])
molden.from_lassi (las, 'c2h4n4_lassirq_631g.molden', si=si_rq)

print ("SI vector (LASSI[{},{}]):".format (r,q))
# print (si_rq)
# util.print_list_matrix(si_rq)
# print("si_rq shape:", si_rq.shape)
print("e_roots, ", e_roots)
# print("e_roots shape:", np.array(e_roots).shape)

lasci_fs = []
# tci = np.transpose(lsi.ci, (1,2,0,3,4)) # frag, roots, eig, alpha, beta -> roots, eig, frag, alpha, beta
ci = lsi.ci
tci = lambda s, c, a: ci[a, s, c]
nroots = len(ci[0])
nfrag = len(ci)


# for root in range(nroots):
#     nelecs = []
#     norbs = []
#     for frag in range(nfrag):
#         nelec_frag = lsi.fciboxes[frag].fcisolvers[root].nelec
#         nelecs.append(nelec_frag)
#         norbs.append(lsi.fciboxes[frag].fcisolvers[root].norb)
    
#     print("root {}".format(root), "norbs =", norbs, "nelecs =", nelecs)
# print("ci: ")
# util.print_list_matrix(ci)
from itertools import product

for root in range(nroots):
    nelecs = []
    norbs = []
    frageigci = []
    for frag in range(nfrag):
        nelec_frag = lsi.fciboxes[frag].fcisolvers[root].nelec
        nelecs.append(nelec_frag)
        norbs.append(lsi.fciboxes[frag].fcisolvers[root].norb)
        
        eigci = ci[frag][root]
        print("eigci.shape = ", eigci.shape)
        if len(eigci.shape) == 2:
            eigci = eigci[np.newaxis, :, :]
        print("After adding newaxis, eigci.shape = ", eigci.shape)
        frageigci.append(eigci)
        
    print("frageigci ")
    util.print_list_matrix(frageigci)
    for prodci in product(*frageigci):
        # print("root {}, norbs {}, nelecs {}, lasci shapes {}".format(root, norbs, nelecs, [c.shape for c in lasci]))
        
        print("root ", root, "norb ", norbs, "nelecas", nelecs)
        print("prodci shapes:", [c.shape for c in prodci])
        print("prodci:")
        util.print_list_matrix(prodci)
        lasci_fs.append(util.cilas2f(prodci, norbs, nelecs))



    # for eig in range(len(ci[frag][root])):
    #     lasci = [ci[i][root][eig] for i in range(nfrag)]
    #     print("root {}, eig {}".format(root, eig))
    #     print("lasci shape = ", [c.shape for c in lasci])
    #     print("norbs =", norbs)
    #     print("nelecs =", nelecs)
    #     lasci_fs.append(util.cilas2f(lasci, norbs, nelecs))


mc_uscc =  mcscf.CASCI (mf, 6, 6)
mc_uscc.mo_coeff = las.mo_coeff
lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, [], [])
mc_uscc.fcisolver.norb_f = [3,3]
fci = mc_uscc.fcisolver


print("lasci_fs length:", len(lasci_fs))

las_psis = []
for las_ci_f in lasci_fs:
    las_psis.append(fci.build_psi (las_ci_f, 6, (3,3), 6))


h1eff, e_core = mc_uscc.get_h1eff (mc_uscc.mo_coeff)
h2eff = mc_uscc.get_h2eff (mc_uscc.mo_coeff)
h = [e_core, h1eff, h2eff]

n = len(las_psis)
lsiS = np.zeros((n,n), dtype=complex)
lsiH = np.zeros((n,n), dtype=complex)

for i in range(n):
    for j in range(n):
        lsiS[i,j], lsiH[i,j] = get_Sij_Hij (las_psis[i], las_psis[j], h)
        print("computing for i,j = ", i, j)

print("LASSI Overlap matrix S:")
util.print_list_matrix (lsiS, digits=4)
print("LASSI Hamiltonian matrix H:")
util.print_list_matrix (lsiH, digits=4)
e_vals_lsi, e_vecs_lsi = linalg.eig (lsiH, lsiS)
idx_lsi = e_vals_lsi.argsort ()
e_vals_lsi = e_vals_lsi[idx_lsi]
e_vecs_lsi = e_vecs_lsi[:,idx_lsi]
print("Diagonalized LASSI energies from psi basis:")
print("State {:.0f} energy = {:.9f}".format(0, e_vals_lsi[0].real))
# print ("SI length (LASSI[{},{}]):".format (r,q), len (si_rq[:,0]))

# # let's try first select only the components larger than 1e-1
# select_threshold = 1e-1
# select_idx = np.ones (len (si_rq[:,0]), dtype=np.int8)
# for i in range (1, len (si_rq[:,0])):
#     if abs (si_rq[i,0]) < select_threshold:
#         #
#         # print("discarding psi[{}] | condition number = {}".format(i - 1, cnum) )
#         select_idx[i - 1] = 0
# print("Selected indices (|SI| >= {}):".format(select_threshold))



