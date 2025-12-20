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


# define the super parameters
r = 1
q = 2
frac = 0.01 # excitation selection fraction


# ------------------------ performing LASSI[r,q] ------------------------ #
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
print("e_roots, ", e_roots)



# ------------------------ performing LASSI[r,q] with LCC ------------------------ #
# first we need to generate the |psi> functions from the LASSI CI vectors
# the LASSI CI are list of 5 indeces: frag, roots, eig, alpha, beta
lasci_fs = []
nroots = len(lsi.ci[0])
nfrag = len(lsi.ci)

from itertools import product
nroots = 3 # for testing
for root in range(nroots):
    nelecs = []
    norbs = []
    frageigci = []
    for frag in range(nfrag):
        nelec_frag = lsi.fciboxes[frag].fcisolvers[root].nelec
        nelecs.append(nelec_frag)
        norbs.append(lsi.fciboxes[frag].fcisolvers[root].norb)
        eigci = lsi.ci[frag][root]
        if len(eigci.shape) == 2:
            eigci = eigci[np.newaxis, :, :]
        frageigci.append(eigci)
    for prodci in product(*frageigci):
        lasci_fs.append(util.cilas2f(prodci, norbs, nelecs))

# now we form psi, including all the excitations
ncas, ncore = las.ncas, las.ncore
nlas = las.ncas_sub
uop = lasuccsd.gen_uccsd_op(ncas,nlas)
a_idxs = uop.a_idxs # for testing
i_idxs = uop.i_idxs

mc_uscc =  mcscf.CASCI (mf, 6, 6)
mc_uscc.mo_coeff = las.mo_coeff
lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs, i_idxs)
mc_uscc.fcisolver.norb_f = [3,3]
fci = mc_uscc.fcisolver

las_psis = []
for las_ci_f in lasci_fs:
    las_psis.append(fci.build_psi (las_ci_f, 6, (3,3), 6))


# Here we diagonalize in the |psi> basis just to confirm it gives the same result as LASSI
h1eff, e_core = mc_uscc.get_h1eff (mc_uscc.mo_coeff)
h2eff = mc_uscc.get_h2eff (mc_uscc.mo_coeff)
h = [e_core, h1eff, h2eff]

n = len(las_psis)
lsiS = np.zeros((n,n), dtype=complex)
lsiH = np.zeros((n,n), dtype=complex)


start = time.time()
for i in range(n):
    for j in range(n):
        lsiS[i,j], lsiH[i,j] = get_Sij_Hij (las_psis[i], las_psis[j], h)
end = time.time()
print("time for building H and S matrices of size {} in psi basis: ".format(n), end - start, " seconds")


print("LASSI Overlap matrix S:")
util.print_list_matrix (lsiS, digits=4)
print("LASSI Hamiltonian matrix H:")
util.print_list_matrix (lsiH, digits=4)
e_vals_lsi, e_vecs_lsi = linalg.eig (lsiH, lsiS)
idx_lsi = e_vals_lsi.argsort ()
e_vals_lsi = e_vals_lsi[idx_lsi]
e_vecs_lsi = e_vecs_lsi[:,idx_lsi]
print("Ground state energy from diagonalizing psi: {:.9f}".format(e_vals_lsi.min().real))
print("The energy difference with LASSI ", e_vals_lsi.min().real - e_roots[0], " should be 0")

# Next we form the gradients with respect to each excitation


# now select only the components with significant contributions
# also we save the fci vectors for the Oi|lsi>

grads = []
dx = 1e-5 # step size for numerical gradient
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

    uiclsi = sum(e_vecs_lsi[i, 0] * uics[i] for i in range(n))
    huiclsi = sum(e_vecs_lsi[i, 0] * huics[i] for i in range(n))    
    e_dx = (uiclsi.conj().dot(huiclsi)) / (uiclsi.conj().dot(uiclsi)) - e_vals_lsi[0]
    grad = np.abs(e_dx.real / dx)
    grads.append(grad)

end = time.time()
print("time for evaluating ", len(a_idxs) , " gradients is ", end - start, " seconds")

# now we obtain the fci vectors for Oi |lsi> for the selected excitations
grad_ordered_indices = np.argsort(grads)[::-1]
n_select = max(1, int(np.ceil(len(a_idxs) * frac)))
selected_indices = grad_ordered_indices[:n_select]


fci_oilsis = []
fci_hoilsis = []
dx = np.pi / 2 # use larger step size for Oi|lsi>
for idx in selected_indices:
    uics = []
    huics = []
    for j in range(n):
        psi = fci.build_psi (lasci_fs[j], 6, (3,3), 6)
        psi.x[psi.nconstr + idx] = dx
        c, uc, huc, uhuc, c_f = psi.hc_x (psi.x, h)
        uics.append(uc.ravel())
        huics.append(huc.ravel())

    uiclsi = sum(e_vecs_lsi[i, 0] * uics[i] for i in range(n))
    huiclsi = sum(e_vecs_lsi[i, 0] * huics[i] for i in range(n))    
    fci_oilsis.append(uiclsi)
    fci_hoilsis.append(huiclsi)

# insert |lsi> itself
for j in range(n):
    psi = fci.build_psi (lasci_fs[j], 6, (3,3), 6)
    # psi.x[psi.nconstr + idx] = dx
    c, uc, huc, uhuc, c_f = psi.hc_x (psi.x, h)
    uics.append(uc.ravel())
    huics.append(huc.ravel())

uiclsi = sum(e_vecs_lsi[i, 0] * uics[i] for i in range(n))
huiclsi = sum(e_vecs_lsi[i, 0] * huics[i] for i in range(n))    
fci_oilsis.append(uiclsi)
fci_hoilsis.append(huiclsi)

m = len(selected_indices)

# diagonalize in the Oi |lsi> basis, including |lsi> itself
Holsi = np.zeros((m+1,m+1), dtype=complex)
Solsi = np.zeros((m+1,m+1), dtype=complex)

start = time.time()
for i in range(m+1):
    for j in range(i, m+1):
        Sij = fci_oilsis[i].conj().dot(fci_oilsis[j])
        Hij = fci_oilsis[i].conj().dot(fci_hoilsis[j])
        Solsi[i,j] = Sij
        Holsi[i,j] = Hij
        if i != j:
            Solsi[j,i] = Sij.conj()
            Holsi[j,i] = Hij.conj()
end = time.time()
print("time for building H and S matrices of size {} in Oi|lsi> basis: ".format(m+1), end - start, " seconds")

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
