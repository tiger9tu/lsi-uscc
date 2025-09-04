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

r = 1
q = 1


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
las.mo_coeff = mo
las.kernel ()
molden.from_lasscf (las, 'c2h4n4_lasscf66_631g.molden')

mc = mcscf.CASCI (mf, 6, 6).set (fcisolver=csf_solver(mol,smult=1))
mc.kernel (las.mo_coeff)
molden.from_mcscf (mc, 'c2h4n4_casscf66_631g.molden', cas_natorb=True)

# print("mc.ci = \n", mc.ci)
print ("LASSCF((3,3),(3,3)) energy =", las.e_tot)
# print ("CASCI(6,6) energy =", mc.e_tot)


ncas = 6
nelecas = 6

# let's use my state interaction
print("las.ci\n", las.ci)

# all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(las, 0.001)

mc_uscc = mcscf.CASCI(mf, ncas, nelecas)
mc_uscc.mo_coeff = las.mo_coeff
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, [], [])
mc_uscc.fcisolver.norb_f = [3,3]
fci = mc_uscc.fcisolver
# easily hit the maximal memory limit
# mc_uscc.fcisolver.frozen = test_config['frozen'] if 'frozen' in test_config else None  
# mc_uscc.kernel(ci0 = cilas2f(las.ci, mol_config['ncas'], mol_config['nelecas']))

# first verify that we can reproduce the energy of LASSI
psis = []

print("nested[0]: \n", to_nested_format(las.ci)[0])

for i in range(len(las.ci)):
    print("las ci[{}] = \n".format(i), las.ci[i])
    
    psis.append(fci.build_psi(cilas2f(las.ci[i], las.ncas_sub, las.nelecas_sub), ncas, las.ncas_sub, nelecas))

h1eff,e_core= mc_uscc.get_h1eff(mc_uscc.mo_coeff)
h2eff = mc_uscc.get_h2eff()
h = [e_core, h1eff, h2eff]

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

# print("las e_tot = ", las_e_tot)

eigvals, eigvecs = eigh(H, S)
print("lowest eigenvalue = ", eigvals[0])
print("lowest eigenvector = \n", eigvecs[:, 0])


print("\n\n############ LASSI #################")
lsi = lassi.LASSIrq(las,r=r,q=q)
e_roots, si_rq = lsi.kernel()

print ("LASSI[{},{}]energy =".format (r,q), e_roots[0])
molden.from_lassi (las, 'c2h4n4_lassirq_631g.molden', si=si_rq)

# print ("SI vector (LASSI[{},{}]):".format (r,q))
# print (si_rq)

print("\n\n############# LASSI TEST ##############")

op = (op_o0, op_o1)
def energy_tot (lsi, mo_coeff=None, ci=None, si=None, soc=0, opt=None):
    if mo_coeff is None: mo_coeff = lsi.mo_coeff
    if ci is None: ci = lsi.ci
    if si is None: si = lsi.si
    print("si size = ", si.shape)
    if opt is None: opt = lsi.opt
    if si.ndim==1:
        nroots_si=1
    else:
        assert (si.ndim==2)
        nroots_si = si.shape[1]
    si = si.reshape (-1,nroots_si)
    nelec_frs = lsi.get_nelec_frs ()
    h0, h1, h2 = lsi.ham_2q (mo_coeff=mo_coeff, soc=soc)
    hop = op[opt].gen_contract_op_si_hdiag (
        lsi, h1, h2, ci, nelec_frs, soc=soc
    )[0]
    e_tot = lib.einsum ('ip,ip->p', (hop (si) + h0*si), si.conj ())
    if nroots_si==1: e_tot=e_tot[0]
    return e_tot

print("LASSI energy =", energy_tot(lsi))