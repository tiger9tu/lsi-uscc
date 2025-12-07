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
from copy import deepcopy
# Using a hand-made model space

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

def get_Sij_Hij(psi_i, psi_j, h):
    ucj, hucj = psi_j.hc_x (psi_j.x, h)[1:3]
    uci, huci = psi_i.hc_x (psi_i.x, h)[1:3]
    ucj, hucj = ucj.ravel (), hucj.ravel ()
    uci = uci.ravel ()
    Sij = uci.conj ().dot (ucj)
    Hij = uci.conj ().dot (hucj)
    return Sij, Hij

n = 4
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


weights = [0.5,0.5,0,0]
spins=[[1,-1],[-1,1],[0,0],[0,0]]
smults=[[2,2],[2,2],[1,1],[1,1]]    
charges=[[0,0],[0,0],[-1,1],[1,-1]]

# nelecas = (sum(las.nelecas_sub[j]) - charges[i][j] + spins[i][j]) // 2
# nelecbs = (sum(las.nelecas_sub[j]) - charges[i][j] - spins[i][j]) // 2


las2 = las.state_average (weights, spins = spins, smults = smults, charges = charges)
las2.lasci ()
las2.dump_spaces ()
print("las2 e_tot = ", las2.e_tot)



lsi = lassi.LASSI(las2)
e_roots, si_hand = lsi.kernel()
print ("LASSI(hand) energy =", e_roots[0])
print ("SI vector (hand):")
print (si_hand[:,0])

lsirdm1s, lsirdm2s = lsi.make_casdm12s ()
lsi_ground_rdm1s = lsirdm1s[0]
lsi_ground_rdm2s = util.lassi_rdm2_to_lasscf (lsirdm2s[0])
g_sel, grad_all, a_idxs, i_idxs = grad.get_grad_exact_rdm12 (las2, lsi_ground_rdm1s, lsi_ground_rdm2s, epsilon=0.0)

# The g_sel is not correct, let's use numerical gradients instead
mc_uscc =  mcscf.CASCI (mf, 6, 6)
mc_uscc.mo_coeff = las2.mo_coeff
lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs, i_idxs)
mc_uscc.fcisolver.norb_f = [3,3]
fci = mc_uscc.fcisolver

nfrag = 2

las_psis = []
las_ci_fs = []
for i in range(n):
    lasi_ci = np.array(las2.ci)[:,i,:,:]
    nelecasi = [(sum(las.nelecas_sub[j]) - charges[i][j] + spins[i][j]) // 2 for j in range(nfrag)]
    nelecbsi = [(sum(las.nelecas_sub[j]) - charges[i][j] - spins[i][j]) // 2 for j in range(nfrag)]
    neleci = [[nelecasi[j], nelecbsi[j]] for j in range(nfrag)]
    print("neleci for state ", i, " = ", neleci)
    lasi_ci_f =  util.cilas2f(lasi_ci, [3,3], neleci)
    psi = fci.build_psi (lasi_ci_f, 6, (3,3), 6)
    las_ci_fs.append(lasi_ci_f)
    las_psis.append(psi)


h1eff, e_core = mc_uscc.get_h1eff (mc_uscc.mo_coeff)
h2eff = mc_uscc.get_h2eff (mc_uscc.mo_coeff)
h = [e_core, h1eff, h2eff]


lsiS = np.zeros((n,n), dtype=complex)
lsiH = np.zeros((n,n), dtype=complex)

for i in range(n):
    for j in range(n):
        lsiS[i,j], lsiH[i,j] = get_Sij_Hij (las_psis[i], las_psis[j], h)

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
print("The energy difference with LASSI ", e_vals_lsi[0].real - e_roots[0], " should be 0")
print("Ground state coefficient vector from psi basis:")
print(e_vecs_lsi[:,0])


dx = 1e-5 # step size for numerical gradient
my_gs = []
for i in range(len(a_idxs)):
    uics = []
    huics = []

    uicsneg = []
    huicsneg = []
    a_idx = a_idxs[i]
    i_idx = i_idxs[i]
    for j in range(n):
        psi = fci.build_psi (las_ci_fs[j], 6, (3,3), 6)
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

a_idxs_selected, i_idxs_selected, g = util.get_sorted_excitations (a_idxs, i_idxs, my_gs, fraction = 0.01, verbose=3)


# confirmed.
# Now let's approch the CASCI limit by including the lcc of each las state
psis = []

m = len(a_idxs)
print("Number of single excitations included: ", m)

# include the initial las states as well
for i in range(n):
    a_idx = a_idxs[j]
    i_idx = i_idxs[j]
    psi = fci.build_psi (las_ci_fs[i], 6, (3,3), 6)
    psis.append(psi)

# for i in range(n):
#     for j in range(m):
#         a_idx = a_idxs[j]
#         i_idx = i_idxs[j]
#         psi = fci.build_psi (las_ci_fs[i], 6, (3,3), 6)
#         psi.x[psi.nconstr + j] = np.pi / 2
#         psis.append(psi)

for j in range(m):
    for i in range(n):
        a_idx = a_idxs[j]
        i_idx = i_idxs[j]
        psi = fci.build_psi (las_ci_fs[i], 6, (3,3), 6)
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
