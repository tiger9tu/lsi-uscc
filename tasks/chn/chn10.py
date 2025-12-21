import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi
from mrh.my_pyscf.tools import molden
from c2h4n4_struct import structure as struct
from pathlib import Path
from mrh.exploratory.citools import grad, lasci_ominus1
from mrh.exploratory.unitary_cc import lasuccsd
import time
from itertools import product
from helper import util
from lcc.lcc_solver import FCISolver_CC
import ast, sys
pwd = Path(__file__).resolve().parent
VERBOSE = 1
# Using LASSI[r,q]

dnn0 = 2.0
dnn1 = 2.0
r = 1
q = 2
frac = 0.01
ncas_f = (4,2,4)
nelecas_f = ((2,2),(1,1),(2,2))
nelecas = tuple(sum(x) for x in zip(*nelecas_f))
spin_sub=(1,1,1)
frag_atom_list = [[0,1,2],[3,4,5,6],[7,8,9]]

grad_path = pwd / 'data' / 'lcc_grad_cas6r1q2.txt'

mol = struct (dnn0, dnn1, '6-31g')
mol.spin = 8
mol.output = pwd / 'data' / 'c2h4n4_lassirq_631g.log'
mol.verbose = lib.logger.INFO
mol.build ()
mf = scf.RHF (mol).run ()

las = LASSCF (mf, ncas_f, nelecas_f, spin_sub=spin_sub)
mo_coeff = las.localize_init_guess ([[0,1,2],[3,4,5,6],[7,8,9]])
las = las.state_average ([0.5,0.5],
    spins=[[1,-1],[-1,1]],
    smults=[[2,2],[2,2]],    
    charges=[[0,0],[0,0]])
las.kernel (mo_coeff)
molden.from_lasscf (las, pwd / 'data' / 'c2h4n4_lasscf66_631g.molden')
mo_coeff = las.mo_coeff

mc = mcscf.CASCI (mf, sum(ncas_f), nelecas).set (fcisolver=csf_solver(mol,smult=1)) # for 
mc.kernel (mo_coeff)
sys.stderr.flush ()
molden.from_mcscf (mc, pwd / 'data' / 'c2h4n4_casscf66_631g.molden', cas_natorb=True)

sys.stderr.flush ()
print (f"LASSCF{ncas_f}, {spin_sub} energy =", las.e_tot)
print (f"CASCI{(sum(ncas_f), nelecas)} energy =", mc.e_tot, flush=True)

lsi = lassi.LASSIrq(las,r=r,q=q)
e_roots, si_rq = lsi.kernel()

print ("LASSI[{},{}]energy =".format (r,q), e_roots[0])
molden.from_lassi (las, pwd / 'data' / 'c2h4n4_lassirq_631g.molden', si=si_rq)

if VERBOSE > 1:
    print ("SI vector (LASSI[{},{}]):".format (r,q))
    print (si_rq[:,0])


# ------------------------ performing LASSI[r,q] with LCC ------------------------ #
uop = lasuccsd.gen_uccsd_op(las.ncas,las.ncas_sub)
a_idxs = uop.a_idxs # for testing
i_idxs = uop.i_idxs

mc_uscc =  mcscf.CASCI (mf, np.sum(ncas_f), np.sum(nelecas_f))
mc_uscc.mo_coeff = las.mo_coeff
lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs, i_idxs)
mc_uscc.fcisolver.norb_f = ncas_f
fci = mc_uscc.fcisolver


# we obtain the las_ci_fs for each las state in lassi
# lsi.ci is a list of shape (frag, roots, eig, alpha, beta)
lasci_fs = []
nroots = len(lsi.ci[0])
nfrag = len(lsi.ci)

# nroots = 3 # for testing
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


# here we confirm that the ground state energy in the fci space is the same as lassi
h1eff, e_core = mc_uscc.get_h1eff ()
h2eff = mc_uscc.get_h2eff ()
h = [e_core, h1eff, h2eff]

las_psis = []
for las_ci_f in lasci_fs:
    las_psis.append(fci.build_psi (las_ci_f, np.sum(ncas_f), ncas_f, nelecas_f))

n = len(las_psis)
lsiS = np.zeros((n,n), dtype=complex)
lsiH = np.zeros((n,n), dtype=complex)

start = time.time()
for i in range(n):
    for j in range(i, n):
        lasi, hlasi = las_psis[i].hc_x (las_psis[i].x, h)[1:3]
        lasj, hlasj = las_psis[j].hc_x (las_psis[j].x, h)[1:3]
        lasi, hlasi = lasi.ravel (), hlasi.ravel ()
        lasj, hlasj = lasj.ravel (), hlasj.ravel ()
        Sij = lasi.conj ().dot (lasj)
        Hij = lasi.conj ().dot (hlasj)
        lsiS[i,j], lsiH[i,j] = Sij, Hij
        if i != j:
            lsiS[j,i] = np.conj(lsiS[i,j])
            lsiH[j,i] = np.conj(lsiH[i,j])

end = time.time()
print("time for building H and S matrices of size {} in psi basis: ".format(n), end - start, " seconds")

if VERBOSE > 2:
    print("LASSI Overlap matrix S:")
    util.print_list_matrix (lsiS, digits=4)
    print("LASSI Hamiltonian matrix H:")
    util.print_list_matrix (lsiH, digits=4)

e_vals_lsi, e_vecs_lsi = linalg.eig (lsiH, lsiS)
idx_lsi = e_vals_lsi.argsort ()
e_vals_lsi = e_vals_lsi[idx_lsi]
e_vecs_lsi = e_vecs_lsi[:,idx_lsi]

if VERBOSE > 3:
    print("Ground state energy from diagonalizing psi: {:.9f}".format(e_vals_lsi.min().real))
    print("The energy difference with LASSI ", e_vals_lsi.min().real - e_roots[0], " should be 0")
    print("Ground state coefficient vector from psi basis:")
    print(e_vecs_lsi[:,0].real)
    print("The difference with LASSI si vector: ")
    print(e_vecs_lsi[:,0].real - si_rq[:,0]) # the sign may differ


# first we obtain gradients of the excitations

try:
    with grad_path.open('r') as f:
        grads = ast.literal_eval(f.read())
except FileNotFoundError:
    grads = None

if grads is None:
    grads = []
    dx = 1e-5 # step size for numerical gradient
    start = time.time()
    for i in range(len(a_idxs)):
        uilass = []
        huilass = []
        a_idx = a_idxs[i]
        i_idx = i_idxs[i]
        for j in range(n):
            psi = lasci_ominus1.LASUCCTrialState (fci, ci0_f, norb, norb_f, nelec)
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
    print("Gradients: ")
    print(grads)
    with grad_path.open('w') as f:
        f.write(str(grads))
    

# print("gradients: \n", grads)


ncc = int(np.ceil(frac * len(grads)))
top_indices = np.argsort(np.abs(grads))[-ncc:][::-1]
print("Selected top {} excitations indices for LCC: \n".format(ncc), top_indices)

a_idxs_sel = [a_idxs[i] for i in top_indices]
i_idxs_sel = [i_idxs[i] for i in top_indices]

# include the |lsi> itself
a_idxs_sel.insert(0, np.array([0], dtype=np.uint8))
i_idxs_sel.insert(0, np.array([0], dtype=np.uint8))

mc_uscc.fcisolver = FCISolver_CC(mol, a_idxs_sel, i_idxs_sel, t = np.pi / 2)
mc_uscc.fcisolver.norb_f = ncas_f
mc_uscc.fcisolver.ci0_fs = lasci_fs
mc_uscc.fcisolver.si = e_vecs_lsi[:,0].real
mc_uscc.kernel()
print("LSILCCSD energy: {:.9f}".format(mc_uscc.e_tot))
print("lccsi vector: ")
print(mc_uscc.fcisolver.lccsi.real)






