# Author: Shreya Verma shreyav@uchicago.edu
# This is a sample script to run LAS-USCCSD for the H4 molecule with the polynomial-scaling algorithm to select cluster excitations
# (2e,2o)+(2e,1o)
# This is not a VQE calculation with statevector simulator, rather the classical emulator is used 

import numpy as np
import pyscf
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad, lasci_ominus1
from scipy.special import comb
from pyscf.fci import cistring

def get_ground_fcivec(norb, nelec):
    """ nelec is number of alpha electrons, assuming the 
    alpha and beta electrons are the same number"""
    k = int(comb(norb, nelec))
    mat = np.zeros((k, k))
    mat[0, 0] = 1.0  # ground state is the first determinant
    return mat

def get_kth_excited_fcivec(norb, nelec, k):
    """ Get the k-th excited state FCI vector """
    k = int(comb(norb, nelec))
    mat = np.zeros((k, k))
    mat[0, 0] = 1.0  # ground state is the first determinant
    return mat


def ci2fock(fci_vec, norb, nelec):
    """
    Transform FCI vector in (na, nb) form into full Fock-space CI vector (2^norb x 2^norb).

    Args:
        fci_vec: np.ndarray, shape (n_alpha_det, n_beta_det)
                 FCI coefficients in PySCF's string-based representation.
        norb: int, number of spatial orbitals
        nelec: tuple or int
               If tuple -> (n_alpha, n_beta)
               If int   -> total electrons (assume closed shell)

    Returns:
        fock: np.ndarray, shape (2**norb, 2**norb)
              CI coefficients in Fock basis representation.
              Row index = alpha occupation bitstring
              Col index = beta occupation bitstring
    """
    if isinstance(nelec, int):
        n_alpha = n_beta = nelec // 2
    else:
        n_alpha, n_beta = nelec

    fock = np.zeros((2**norb, 2**norb), dtype=fci_vec.dtype)

    # list of bitstring integers for alpha and beta sectors
    alpha_strs = cistring.make_strings(range(norb), n_alpha)
    beta_strs  = cistring.make_strings(range(norb), n_beta)

    for ia, a_occ in enumerate(alpha_strs):
        for ib, b_occ in enumerate(beta_strs):
            fock[a_occ, b_occ] = fci_vec[ia, ib]
    
    return fock


# Initializing the molecule with RHF
#===================================
xyz = '''H 0.0 0.0 0.0;
            H 1.0 0.0 0.0;
            H 0.2 1.6 0.1;
            H 1.159166 1.3 -0.1
'''
mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log.py',
    verbose=0)
mf = scf.RHF (mol).run ()

print("RHF energy = ", mf.e_tot)

# ref = mcscf.CASSCF (mf, 4, 4).run () # = FCI

# Running LASSCF
#===================================
las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
las.verbose = 4
frag_atom_list = ((0,1),(2,3))
mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)

print("LASSCF energy = ", las.e_tot)

mc_uscc = mcscf.CASCI(mf, 4, 4)
mc_uscc.mo_coeff = las.mo_coeff
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, [], [])
norb = mf.mo_coeff.shape[1]
norb_f = [norb]
nelec = mol.nelectron

fcisolver = mc_uscc.fcisolver


ground_fcivec = get_ground_fcivec(norb, nelec // 2)      
fock_vec0 = ci2fock(ground_fcivec, norb, nelec)
print("fock vec.shape = ", fock_vec0.shape)

psi0 =  getattr (fcisolver, 'psi', fcisolver.build_psi ([fock_vec0], norb, norb_f, nelec))
h1eff,e_core= mc_uscc.get_h1eff(mc_uscc.mo_coeff)
h2eff = mc_uscc.get_h2eff()
h = [e_core, h1eff, h2eff]

energy = psi0.energy_tot(psi0.x, h)
print("mc-uscc not optimized energy = ", energy)

