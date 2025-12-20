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
import time
# Using LASSI[r,q]

def get_Sij_Hij(psi_i, psi_j, h):
    ucj, hucj = psi_j.hc_x (psi_j.x, h)[1:3]
    uci, huci = psi_i.hc_x (psi_i.x, h)[1:3]
    ucj, hucj = ucj.ravel (), hucj.ravel ()
    uci = uci.ravel ()
    Sij = uci.conj ().dot (ucj)
    Hij = uci.conj ().dot (hucj)
    return Sij, Hij

def select_psi(S, cond_thresh=8e5):
    n = S.shape[0]
    select_idx = np.ones(n, dtype=bool)
    for i in range(1, n + 1):
        Si = S[:i, :i]
        try:
            cnum = np.linalg.cond(Si)
        except Exception:
            cnum = float('inf')
        if cnum > cond_thresh:
            # print("discarding psi[{}] | condition number = {}".format(i - 1, cnum) )
            select_idx[i - 1] = 0
    return select_idx


r = 1
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
# debug, use only first 3 roots
for root in range(nroots):
    nelecs = []
    norbs = []
    frageigci = []
    for frag in range(nfrag):
        nelec_frag = lsi.fciboxes[frag].fcisolvers[root].nelec
        nelecs.append(nelec_frag)
        norbs.append(lsi.fciboxes[frag].fcisolvers[root].norb)
        
        eigci = ci[frag][root]
        if len(eigci.shape) == 2:
            eigci = eigci[np.newaxis, :, :]
        frageigci.append(eigci)
        
    util.print_list_matrix(frageigci)
    for prodci in product(*frageigci):
        # print("root {}, norbs {}, nelecs {}, lasci shapes {}".format(root, norbs, nelecs, [c.shape for c in lasci]))
        util.print_list_matrix(prodci)
        lasci_fs.append(util.cilas2f(prodci, norbs, nelecs))



    # for eig in range(len(ci[frag][root])):
    #     lasci = [ci[i][root][eig] for i in range(nfrag)]
    #     print("root {}, eig {}".format(root, eig))
    #     print("lasci shape = ", [c.shape for c in lasci])
    #     print("norbs =", norbs)
    #     print("nelecs =", nelecs)
    #     lasci_fs.append(util.cilas2f(lasci, norbs, nelecs))
ncas, ncore = las.ncas, las.ncore
# Generate indices
nlas = las.ncas_sub
uop = lasuccsd.gen_uccsd_op(ncas,nlas)
a_idxs = uop.a_idxs
i_idxs = uop.i_idxs

mc_uscc =  mcscf.CASCI (mf, 6, 6)
mc_uscc.mo_coeff = las.mo_coeff
lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs, i_idxs)
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

# now select only the components with significant contributions
num_psi = 8
sorted_indices = np.argsort(np.abs(e_vecs_lsi[:,0]))[::-1]
selected_indices = sorted_indices[:num_psi]
print("Selected psi indices for ground state (top {}): {}".format(num_psi, selected_indices))
print("Ground state coefficient vector from psi basis:")
print(e_vecs_lsi[:,0])
# diagnoalize in this selected basis
S_selected = lsiS[np.ix_(selected_indices, selected_indices)]
H_selected = lsiH[np.ix_(selected_indices, selected_indices)]
e_vals_selected, e_vecs_selected = linalg.eig(H_selected, S_selected)
idx_selected = e_vals_selected.argsort()
e_vals_selected = e_vals_selected[idx_selected]
e_vecs_selected = e_vecs_selected[:, idx_selected]
print("Diagonalized LASSI energies from selected psi basis:")
print("State {:.0f} energy = {:.9f}".format(0, e_vals_selected[0].real))
print("Ground state coefficient vector from selected psi basis:")
print(e_vecs_selected[:,0])

selected_psis = [las_psis[i] for i in selected_indices]
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


dx = 1e-5 # step size for numerical gradient
my_gs = []
t = 10
print("len a_idxs ", len(a_idxs))

start = time.time()

for i in range(len(a_idxs)):
    uics = []
    huics = []

    uicsneg = []
    huicsneg = []
    a_idx = a_idxs[i]
    i_idx = i_idxs[i]
    for j in range(n):
        psi = fci.build_psi (lasci_fs[j], 6, (3,3), 6)
        psi.x[psi.nconstr + i] = dx
        c, uc, huc, uhuc, c_f = psi.hc_x (psi.x, h)
        uics.append(uc.ravel())
        huics.append(huc.ravel())

        psi.x[psi.nconstr + i] = -dx
        cneg, ucneg, hucneg, uhucneg, c_fneg = psi.hc_x (psi.x, h)
        uicsneg.append(ucneg.ravel())
        huicsneg.append(hucneg.ravel())

    uiclsi = sum(e_vecs_lsi[i, 0] * uics[i] for i in range(n))
    huiclsi = sum(e_vecs_lsi[i, 0] * huics[i] for i in range(n))    
    e_dx = (uiclsi.conj().dot(huiclsi)) / (uiclsi.conj().dot(uiclsi)) - e_vals_lsi[0]

    uiclsineg = sum(e_vecs_lsi[i, 0] * uicsneg[i] for i in range(n))
    huiclsineg = sum(e_vecs_lsi[i, 0] * huicsneg[i] for i in range(n))
    e_dx_neg = (uiclsineg.conj().dot(huiclsineg)) / (uiclsineg.conj().dot(uiclsineg)) - e_vals_lsi[0]
    # print("Numerical gradient step ", i, " : pos", e_dx.real / dx, " neg", e_dx_neg.real / dx, " g ", (e_dx - e_dx_neg) / (2 * dx), " compared to analytical ", g[i])
    my_gs.append(e_dx.real)


end = time.time()
print("time for evaluating ", t , " gradients is ", end - start, " seconds")
a_idxs_selected, i_idxs_selected, g = util.get_sorted_excitations (a_idxs, i_idxs, my_gs, fraction = 0.01, verbose=3)


# confirmed.
# Now let's approch the CASCI limit by including the lcc of each las state
psis = selected_psis

m = len(a_idxs_selected)
print("Number of single excitations included: ", m)

print("Selected a_idxs: ", a_idxs_selected)
print("Selected i_idxs: ", i_idxs_selected)

# include the initial las states as well
# for i in range(n):
#     psi = fci.build_psi (lasci_fs[i], 6, (3,3), 6)
#     psis.append(psi)


for j in range(m):
    for i in range(n):
        a_idx = a_idxs_selected[j]
        i_idx = i_idxs_selected[j]
        psi = fci.build_psi (lasci_fs[i], 6, (3,3), 6)
        psi.x[psi.nconstr + j] = np.pi / 2
        psis.append(psi)


n2 = len (psis)
S = np.zeros ((n2,n2), dtype=complex)
H = np.zeros ((n2,n2), dtype=complex)
for i in range(n2):
    for j in range(n2):
        S[i,j], H[i,j] = get_Sij_Hij (psis[i], psis[j], h)
print("Overlap matrix S with LCC:")
util.print_list_matrix (S , digits=4)
print("Hamiltonian matrix H with LCC:")
util.print_list_matrix (H , digits=4)
e_vals, e_vecs = linalg.eig (H, S)
idx = e_vals.argsort ()
e_vals = e_vals[idx]
e_vecs = e_vecs[:,idx]
print("Diagonalized LASSI energies with LCC:")
print("State {:.0f} energy = {:.9f}".format(0, e_vals[0].real))

cond_S = np.linalg.cond(S)
print("Condition number of S:", cond_S)
print("Ground state coefficient vector:")
print(e_vecs[:,0])

# now we filter the psis based on condition number of S
select_idx = select_psi (S, cond_thresh=8e5)
print("Selected indices after filtering based on condition number:")
print(select_idx)
S_selected = S[np.ix_(select_idx, select_idx)]
H_selected = H[np.ix_(select_idx, select_idx)]
e_vals_selected, e_vecs_selected = linalg.eig(H_selected, S_selected)
idx_selected = e_vals_selected.argsort()
e_vals_selected = e_vals_selected[idx_selected]
e_vecs_selected = e_vecs_selected[:, idx_selected]
print("Ground state energy (selected): {:.9f}".format(e_vals_selected[0].real))
print("Ground state coefficients (selected):")
print(e_vecs_selected[:, 0])


# diagonalize in the Oi |lsi> basis
# using the H and S matrices 
Holsi = np.zeros((m+1,m+1), dtype=complex)
Solsi = np.zeros((m+1,m+1), dtype=complex)
for i in range(m+1):
    for j in range(m+1):
        Hoij = 0
        Soij = 0
        for k in range(n):
            for l in range(n):
                coeff_k = e_vecs_lsi[k,0]
                coeff_l = e_vecs_lsi[l,0]
                Hoij += coeff_k.conj () * coeff_l * H[k + i*n, l + j*n]
                Soij += coeff_k.conj () * coeff_l * S[k + i*n, l + j*n]

        Holsi[i,j] = Hoij
        Solsi[i,j] = Soij    


print("Hosi: ")
util.print_list_matrix (Holsi , digits=4)
print("Sosi: ")
util.print_list_matrix (Solsi , digits=4)

e_vals_olsi, e_vecs_olsi = linalg.eig(Holsi, Solsi)
idx_olsi = e_vals_olsi.argsort()
e_vals_olsi = e_vals_olsi[idx_olsi]
e_vecs_olsi = e_vecs_olsi[:, idx_olsi]
print("Ground state energy (Olsi basis): {:.9f}".format(e_vals_olsi[0].real))
print("Ground state coefficients (Olsi basis):")
print(e_vecs_olsi[:, 0])
