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
from mrh.exploratory.citools import grad, lasci_ominus1, fockspace


from scipy import linalg, optimize

def kernel (fci, h1, h2, norb, nelec, norb_f=None, ci0_f=None,
            tol=1e-8, gtol=1e-6, max_cycle=None, 
            orbsym=None, wfnsym=None, ecore=0, **kwargs):
    if max_cycle is None:
        max_cycle = GLOBAL_MAX_CYCLE if GLOBAL_MAX_CYCLE is not None else 15000
    if norb_f is None: norb_f = getattr (fci, 'norb_f', [norb])
    if ci0_f is None: ci0_f = fci.get_init_guess (norb, nelec, norb_f, h1, h2)

    frozen = getattr (fci, 'frozen', None)
    psi = getattr (fci, 'psi', fci.build_psi (ci0_f, norb, norb_f, nelec,
        log=log, frozen=frozen))
    assert (psi.check_ci0_constr)
    psi_options = {'gtol':     gtol,
                   'maxiter':  max_cycle,
                   'disp':     verbose>lib.logger.DEBUG}
    h = [ecore, h1, h2]
    psi_callback = psi.get_solver_callback (h)
    res = optimize.minimize (psi.e_de, psi.x, args=(h,), method='BFGS',
        jac=True, callback=psi_callback, options=psi_options)

    fci.converged = res.success
    e_tot = psi.energy_tot (res.x, h)
    ci1 = psi.get_fcivec (res.x)
    psi.x = res.x
    psi.converged = res.success
    psi.finalize_()
    fci.psi = psi
    return e_tot, ci1


def my_e_de(x, h, position, psi):
    all_x = psi.x.copy()
    all_x[position] = x
    e_tot, all_jac = psi.e_de(all_x, h)
    return e_tot, all_jac[position]



def cilas2f(lasci, norb_f, nelec_f):
    ''' nelec (na, nb)'''
    ci_f = []
    for i, ci in enumerate(lasci):
        ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
    return ci_f

def get_fragment_pair_excitations(all_a_idxs, all_i_idxs, frag_pair, frag_spin_orbs):
    """
    Extract excitations within a specific fragment pair
    
    Parameters:
    -----------
    all_a_idxs, all_i_idxs : List
        All available excitation indices
    frag_pair : Tuple[int, int]
        Fragment pair indices (e.g., (0,1) for fragments 0 and 1)
        
    Returns:
    --------
    Tuple[List, List]
        Filtered excitation indices for the fragment pair
    """
    # Get combined spin orbitals for the fragment pair
    frag_orbs = set(frag_spin_orbs[frag_pair[0]] + frag_spin_orbs[frag_pair[1]])

    pair_a_idxs = []
    pair_i_idxs = []
    for idx in range(len(all_a_idxs)):
        if (
            all(elem in frag_orbs for elem in all_a_idxs[idx]) and
            all(elem in frag_orbs for elem in all_i_idxs[idx])
        ):
            pair_a_idxs.append(all_a_idxs[idx])
            pair_i_idxs.append(all_i_idxs[idx])
    return pair_a_idxs, pair_i_idxs
    


# Initializing the molecule with RHF
#===================================
xyz = ''' H      0.000000000000   0.000000000000   0.000000000000
    H      1.000000000000   0.000000000000   0.000000000000
    H      0.273746762116   2.195450598147   0.100000000000
    H      1.232912762116   1.895450598147  -0.100000000000
    H      0.507178110854   4.193780995243   0.049334760036
    H      1.506140937609   3.988021397347  -0.049334760036
    '''

ncas_f = [2,2,2]
nelec_f = [2,2,2]
spin_f = [1,1,1]

ncas = sum(ncas_f)
nelec = sum(nelec_f)

mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log.py',
    verbose=0)
mf = scf.RHF (mol).run ()
ref = mcscf.CASSCF (mf, ncas, nelec).run () # = FCI
print("CASCF energy (6,6): {:.9f}".format(ref.e_tot))

# Running LASSCF
#===================================
las = LASSCF (mf, ncas_f, nelec_f, spin_sub=spin_f)
frag_atom_list = ((0,1),(2,3),(4,5))

mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)

print("las energy = ", las.e_tot)


#Getting gradient for all cluster excitations through LAS-UCCSD gradients, may use your desired epsilon for selection
#====================================================================================================================
epsilon=0.001
all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(las, epsilon=epsilon)



frag_spin_orb = {0:[0,1,6,7], 1:[2,3,8,9], 2:[4,5,10,11]}

frag_pairs = ((0,1), (1,2))  
n_vqe = len(frag_pairs)

a_idxs = []
i_idxs = []

frag_pairs_sep_idxs = []

for frag_pair in frag_pairs:
    frag_pair_a_idxs, frag_pair_i_idxs = get_fragment_pair_excitations(a_idxs_selected, i_idxs_selected, frag_pair, frag_spin_orb)
    a_idxs.extend(frag_pair_a_idxs)
    i_idxs.extend(frag_pair_i_idxs)
    frag_pairs_sep_idxs.append(len(a_idxs))


ci0_f = cilas2f(las.ci, ncas_f, nelec_f)

print("ci0_f = \n", ci0_f)

mc_uscc = mcscf.CASCI(mf, ncas, nelec)
mc_uscc.mo_coeff = las.mo_coeff
lasci_ominus1.GLOBAL_MAX_CYCLE = 15
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs, i_idxs)
mc_uscc.fcisolver.norb_f = ncas_f

fci = mc_uscc.fcisolver

h1eff,e_core= mc_uscc.get_h1eff(mc_uscc.mo_coeff)
h2eff = mc_uscc.get_h2eff()
h = [e_core, h1eff, h2eff]

psi = getattr (fci, 'psi', fci.build_psi (ci0_f, ncas, ncas_f, nelec))

print("psi energy = ", psi.energy_tot(psi.x, h))

print("number of excitations = ", len(a_idxs))
print("len psi.x = ", len(psi.x))
print("nconstr = ", psi.nconstr)
print("uop.ngen_uniq = ", psi.uop.ngen_uniq)
print("n ci rotation paramter = ", len(psi.x) - (psi.uop.ngen_uniq + psi.nconstr))

print("frag_pairs_sep_idxs = ", frag_pairs_sep_idxs)

for i in range(len(frag_pairs)):
    frag_pair = frag_pairs[i]
    x_positions = []
    x_positions.extend(0) # constr paramter

    uop_x_start = 1 if i == 0 else frag_pairs_sep_idxs[i-1] + 1
    uop_x_end = frag_pairs_sep_idxs[i] + 1
    x_positions.extend(range(uop_x_start, uop_x_end)) # uop paramter


# print("xci size = ", sum ([c.size for c in psi.ci_f]))
# print("psi.ci_f = ", psi.ci_f)
# def nvar_tot (self):
#     return self.nconstr + self.uop.ngen_uniq + sum ([c.size for c in self.ci_f])


# for i, (a_idxs, i_idxs) in enumerate(zip(frag_pairs_a_idxs, frag_pairs_i_idxs)):
#     mc_uscc = mcscf.CASCI(mf, ncas, nelec)
#     mc_uscc.mo_coeff = las.mo_coeff
#     lasci_ominus1.GLOBAL_MAX_CYCLE = 15
#     mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs, i_idxs)
#     mc_uscc.fcisolver.norb_f = ncas_f
    # e_tot, e_cas, ci, mo_coeff, mo_energy = mc_uscc.kernel(ci0=ci_iter)
    # print("optimize iteration: ", i, " energy = ", e_tot)
    # # print("ci = ", ci)
    # print("ci shape = ", np.shape(ci))


