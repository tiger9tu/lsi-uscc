import numpy as np
import pyscf
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad, lasci_ominus1, fockspace
import time
from typing import Iterable, Union

from scipy.linalg import eigh

def flatten(seq: Iterable) -> list:
    result = []
    for item in seq:
        if isinstance(item, (list, tuple, np.ndarray)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def cilas2f(lasci, norb_f, nelec_f):
    ''' nelec (na, nb)'''
    ci_f = []
    for i, ci in enumerate(lasci):
        ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
    return ci_f

def get_Sij_Hij(psi_i, psi_j, h):
    ucj, hucj = psi_j.hc_x (psi_j.x, h)[1:3]
    uci, huci = psi_i.hc_x (psi_i.x, h)[1:3]
    ucj, hucj = ucj.ravel (), hucj.ravel ()
    uci = uci.ravel ()
    Sij = uci.conj ().dot (ucj)
    Hij = uci.conj ().dot (hucj)
    return Sij, Hij

def print_matrix(mat):
    for row in mat:
        print("  ".join(f"{x:.17f}" for x in row))


# Initializing the molecule with RHF
#===================================
xyz = '''H 0.0 0.0 0.0;
            H 1.0 0.0 0.0;
            H 0.2 1.6 0.1;
            H 1.159166 1.3 -0.1'''
mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log.py',
    verbose=0)
mf = scf.RHF (mol).run ()
ref = mcscf.CASSCF (mf, 4, 4).run () # = FCI


# Running LASSCF
#===================================
las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
# las.verbose = 4
# las.nroots = 2
frag_atom_list = ((0,1),(2,3))
mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.bin_excite = True
# las.state_average_(weights=[0.5,0.25,0.25], spins=[[0,0],[1,-1],[-1,-1]]) # not sure about the spin
print("frag atom = ", frag_atom_list)
print("mf.mo_coeff = ", mf.mo_coeff)
print("mo_loc = ", mo_loc)
las_e_tot, las_e_states, las_ci, las_mo_coeff, las_mo_energy, las_h2eff_sub, las_veff = las.kernel (mo_loc)

print("las e_tot = ", las_e_tot)
if las.bin_excite:
    las.ci = las.ci[0] # notice this 

# print("las.ci = ", las_ci)
print("las e_states = ", las_e_states)
print("las e_tot = ", las_e_tot)
# print("las ci = ", las_ci)
print("las ci[0] = \n", las_ci[0])
print("las ci[1] = \n", las_ci[1])
all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(las, epsilon=0.01)

mc_uscc = mcscf.CASCI(mf, 4, 4)
mc_uscc.mo_coeff = las.mo_coeff

mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
mc_uscc.fcisolver.norb_f = [2,2]
mc_uscc.kernel(ci0 = cilas2f(las.ci, las.ncas_sub, las.nelecas_sub))



fci = mc_uscc.fcisolver
x = fci.psi.x

ncas = sum(flatten(las.ncas_sub))
nelec = sum(flatten(las.nelecas_sub))

psis = []
nfrag = len(las.ncas_sub)
for i in range(2**nfrag):
    # Get binary string of i, padded to nfrag length
    bin_str = format(i, f'0{nfrag}b')
    ci = []
    for frag_idx, bit in enumerate(bin_str):
        if bit == '1':
            ci.append(cilas2f(las_ci[1], las.ncas_sub, las.nelecas_sub)[frag_idx])
        else:
            ci.append(cilas2f(las_ci[0], las.ncas_sub, las.nelecas_sub)[frag_idx])
    psis.append(fci.build_psi(ci, ncas, las.ncas_sub, nelec))
    psis[-1].x = x


# las_fock_ci1 = cilas2f(las_ci[0], las.ncas_sub, las.nelecas_sub)
# las_fock_ci2 = cilas2f(las_ci[1], las.ncas_sub, las.nelecas_sub)

# psi1 = fci.build_psi (las_fock_ci1, ncas, las.ncas_sub, nelec) # not sure about this
# psi2 = fci.build_psi (las_fock_ci2, ncas, las.ncas_sub, nelec) # not sure about this
# psi1.x = x
# psi2.x = x


# psis = [psi1, psi2]



h1eff,e_core= mc_uscc.get_h1eff(mc_uscc.mo_coeff)
h2eff = mc_uscc.get_h2eff()

# h1eff,e_core= las.get_h1eff(las.mo_coeff)
# h2eff = las.get_h2eff()

h = [e_core, h1eff, h2eff]

nc = len(psis)

S = np.zeros((nc, nc), dtype=np.complex128)
H = np.zeros((nc, nc), dtype=np.complex128)


for i in range(nc):
    for j in range(nc):
        S[i, j], H[i, j] = get_Sij_Hij(psis[i], psis[j], h)


# print("uc1 = ", psi1.u)
# print("uc2 = ", psi2.u)
print("S matrix \n")
print_matrix(S)
print("H matrix \n")
print_matrix(H)

print("las e_tot = ", las_e_tot)

eigvals, eigvecs = eigh(H, S)
print("lowest eigenvalue = ", eigvals[0])
print("lowest eigenvector = \n", eigvecs[:, 0])

