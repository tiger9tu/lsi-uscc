import numpy as np
import pyscf
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad, lasci_ominus1
from lcc.lcc_solver import FCISolver_CC
from helper.util import print_list_matrix, get_sorted_excitations, cilas2f

from pathlib import Path
def print_sparse_ci(ci, threshold=1e-5):
    for i in range(len(ci)):
        if abs(ci[i]) > threshold:
            num_digits = len(bin(len(ci) - 1)) - 2
            print(f"ci[{i}] = {ci[i]:.6f}, i (binary) = {bin(i)[2:].zfill(num_digits)}")
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
print ("RHF energy = ", mf.e_tot)

# Running LASSCF
#===================================
las = LASSCF (mf, ncas_f, nelecas_f, spin_sub=spin_sub_f, verbose=3)
mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)
print ("LASSCF energy = ", las.e_tot)
las_ci0_f = cilas2f(las.ci, ncas_f, nelecas_f)


only_a_idx = [5, 1] 
only_i_idx = [6, 0]
# a0'a1' i1'i0'
# First perform jordan-wigner transformation implicitly to get the final per-site operators
nspin_orbs = 8
res_vaccum = [1 for _ in range(nspin_orbs)] # 1 -> unoccupied, 2 -> occupied, +- sign, +-3 -> vanish
res_occupied =[2 for _ in range(nspin_orbs)]

for i_idx in only_i_idx: # i1 i0 -> apply i0 then i1
    # ik = s0..sk-1 bk
    for res in [res_vaccum, res_occupied]:
        for k in range(i_idx):
            # s0,..,sk-1
            phase = -1 if abs(res[k]) == 2 else 1
            res[k] *= phase
        # bk
        if abs(res[i_idx]) == 2:
            res[i_idx] /= 2
        else:
            res[i_idx] = 3

for a_idx in only_a_idx:
    # ak' = s0..sk-1 ck'
    for res in [res_vaccum, res_occupied]:
        for k in range(a_idx):
            # s0,..,sk-1
            phase = -1 if abs(res[k]) == 2 else 1
            res[k] *= phase
        # ck'
        if abs(res[a_idx]) == 1:
            res[a_idx] *= 2
        else:
            res[a_idx] = 3
from pyscf.fci import cistring

def binarr(index, length):
    return [int(b) for b in format(index, f'0{length}b')]

frag_orbs = [[0, 1], [2, 3]]
frag_sorbs = [[0,1,4,5], [2,3,6,7]] # we have to generate frag_sorbs from frag_orbs
nelec_f = [2,2]
# we simply apply the fragmented operator on each fragment and then do the inner product

inner_prod = 1
for i, ci_f in enumerate(las_ci0_f):
    new_ci_f = np.zeros_like(ci_f.ravel())
    for det_idx, amp in enumerate(ci_f.ravel()):
        if abs(amp) < 1e-5:
            continue
        binary_array = binarr(det_idx, len(frag_sorbs[i]))
        res_bin_array = np.zeros_like(binary_array)
        coef = amp
        for forb_idx, occ in enumerate(binary_array[::-1]):
            orb_idx = frag_sorbs[i][forb_idx]
            if occ == 0:
                res = res_vaccum[orb_idx]
            else:
                res = res_occupied[orb_idx]
            if res == 3:
                coef = 0
                break
            if abs(res) == 1:
                res_bin_array[forb_idx] = 0
            elif abs(res) == 2:
                res_bin_array[forb_idx] = 1
            coef *= -1 if res < 0 else 1
        
        res_det_idx = int(''.join(map(str, res_bin_array[::-1])), 2)
        new_ci_f[res_det_idx] += coef
    
    inner_prod *= np.vdot(ci_f.ravel(), new_ci_f.ravel())

print("Inner product with fragmented operator: ", inner_prod)