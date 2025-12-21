import numpy as np
import pyscf
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad, lasci_ominus1
# from lcc.lcc_solver import FCISolver_CC
from helper.util import print_list_matrix, get_sorted_excitations, cilas2f

from pathlib import Path

pwd = Path(__file__).resolve().parent
geom_path = pwd.parent / 'geom' / 'h4.xyz'
with geom_path.open('r') as f:
    xyz = f.read()

# Initializing the molecule with RHF
#===================================
ncas_f = (2,2)
nelecas_f = (2,2)
spin_sub_f = (1,1)
frag_atom_list = ((0,1),(2,3))

mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log',
    verbose=0)
mf = scf.RHF (mol).run ()
ref = mcscf.CASCI (mf, 4, 4).run () # = FCI
print ("RHF energy = ", mf.e_tot)
print ("CASCI energy = ", ref.e_tot)


# Running LASSCF
#===================================
las = LASSCF (mf, ncas_f, nelecas_f, spin_sub=spin_sub_f)
las.verbose = 4

mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)
print ("LASSCF energy = ", las.e_tot)
las_ci0_f = cilas2f(las.ci, ncas_f, nelecas_f)
#Getting gradient for all cluster excitations through LAS-UCCSD gradients, may use your desired epsilon for selection
#====================================================================================================================

epsilon = 0.0
all_g, g_sel, a_idxs, i_idxs = grad.get_grad_exact(las, epsilon)
# sort the selected excitations by gradiDent magnitude
gredients = np.array(g_sel)[:,0]
sorted_indices = np.argsort(-np.abs(gredients))
# a_idxs = [a_idxs[i] for i in sorted_indices]
# i_idxs = [i_idxs[i] for i in sorted_indices]


#Computing energy through the LAS-UCC kernel using selected excitations
#==========================================================================================
# epsilon=0.01
mc_uscc = mcscf.CASCI(mf, sum(ncas_f), sum(nelecas_f))
mc_uscc.mo_coeff = las.mo_coeff
lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs, i_idxs)
mc_uscc.fcisolver.norb_f = [2,2]
# mc_uscc.kernel(ci0=las_ci0_f)
# print("Epsilon: {:.9f} | Number of parameters: {:.0f} | LASUSCCSD energy: {:.9f}".format(epsilon, len(a_idxs), mc_uscc.e_tot))

# # print("a_idxs = ", a_idxs)
# # to include the Identity operator
# # a_idxs = []
# # i_idxs = []
# a_idxs.insert(0, np.array([0], dtype=np.uint8))
# i_idxs.insert(0, np.array([0], dtype=np.uint8))
# print("a_idxs = ", a_idxs)
# t does not matter now, just to avoid error
# mc_uscc.fcisolver = FCISolver_CC(mol, a_idxs, i_idxs, t = 1000)

# mc_uscc.fcisolver.norb_f = ncas_f

# mc_uscc.kernel(ci0=las_ci0_f)
# print("LASLCCSD energy: {:.9f}".format(mc_uscc.e_tot))

# # c = mc_uscc.fcisolver.psi0.dp_ci (las_ci0_f)
# # print("norm c = ", np.linalg.norm(c))
h1eff, e_core = mc_uscc.get_h1eff (mc_uscc.mo_coeff)
h2eff = mc_uscc.get_h2eff (mc_uscc.mo_coeff)
h = [e_core, h1eff, h2eff]

las_ci_fs = [las_ci0_f]
coeffs = [1.0]
e0 = las.e_tot
dx = 1e-5 # step size for numerical gradient
n = 1
fci = mc_uscc.fcisolver
# all_g, g_sel, a_idxs, i_idxs = grad.get_grad_exact(las, 0.0)

def get_numerical_gradients(a_idxs, i_idxs, las_ci_fs, coeffs, e0, h, dx):
    my_gs = []
    for i in range(len(a_idxs)):
        uics = []
        huics = []

        uicsneg = []
        huicsneg = []
        a_idx = a_idxs[i]
        i_idx = i_idxs[i]
        for j in range(n):
            psi = fci.build_psi (las_ci_fs[j], 4, (2,2), 4)
            psi.x[psi.nconstr + i] = dx
            c, uc, huc, uhuc, c_f = psi.hc_x (psi.x, h)
            uics.append(uc.ravel())
            huics.append(huc.ravel())

            psi.x[psi.nconstr + i] = -dx
            cneg, ucneg, hucneg, uhucneg, c_fneg = psi.hc_x (psi.x, h)
            uicsneg.append(ucneg.ravel())
            huicsneg.append(hucneg.ravel())

        uiclsi = sum(coeffs[i] * uics[i] for i in range(n))
        huiclsi = sum(coeffs[i] * huics[i] for i in range(n))    
        e_dx = (uiclsi.conj().dot(huiclsi)) / (uiclsi.conj().dot(uiclsi)) - e0

        uiclsineg = sum(coeffs[i] * uicsneg[i] for i in range(n))
        huiclsineg = sum(coeffs[i] * huicsneg[i] for i in range(n))
        e_dx_neg = (uiclsineg.conj().dot(huiclsineg)) / (uiclsineg.conj().dot(uiclsineg)) - e0
        # print("Numerical gradient step ", i, " : pos", e_dx.real / dx, " neg", e_dx_neg.real / dx, " g ", (e_dx - e_dx_neg) / (2 * dx), " compared to analytical ", g[i])
        my_gs.append(e_dx.real / dx)
    return my_gs
gredients = np.array(g_sel)[:,0]
num_gs = get_numerical_gradients(a_idxs, i_idxs, las_ci_fs, coeffs, e0, h, dx)

top_10_indices = np.argsort(-np.abs(gredients))[:10]
print("Analytical gradients (top 10):", gredients[top_10_indices])

top_10_indices_num = np.argsort(-np.abs(num_gs))[:10]
print("Numerical gradients (top 10):", np.array(num_gs)[top_10_indices_num])

print("analytical top 10 indexes vs numerical top 10 indexes:")
print("analytical:", top_10_indices)
print("numerical:", top_10_indices_num)