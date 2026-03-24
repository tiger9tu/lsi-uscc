"""Quick smoke test: single-root LASSCF, 2 LASSIS reference states, 2 excitations."""
import numpy as np
from pyscf import gto, scf, lib
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.exploratory.citools import grad
from mrh.my_pyscf import lassi
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from lcc import LSI_LUSCC
from helper.util import get_sorted_excitations

SI_THRESHOLD = 0.4  # raise to select only ~2 dominant LASSIS components as references

ncas_f         = (4, 2, 4)
nelecas_f      = ((2, 2), (1, 1), (2, 2))
spin_sub_f     = (1, 1, 1)
frag_atom_list = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]

mol = struct(2.0, 2.0, '6-31g')
mol.verbose = 3
mol.spin = 8
mol.max_memory = 8000
mol.build()
mf = scf.RHF(mol).run()

# Single-root LASSCF (multi-root LASSCF is not supported by LASSIS/grad pipeline)
las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f)
mo_coeff = las.localize_init_guess(frag_atom_list)
las.kernel(mo_coeff)
print("LASSCF energy =", las.e_tot)

# LASSIS
lsi = lassi.LASSIS(las)
e_lsi, si = lsi.kernel()
print("LASSIS ground state energy =", e_lsi[0])
print("si.shape =", si.shape)

# 2 excitations (top-2 by gradient)
_, g_sel, a_idxs_all, i_idxs_all = grad.get_grad_exact(las, epsilon=0.0)
g_all = np.array(g_sel)[:, 0]
a_idxs, i_idxs, _ = get_sorted_excitations(a_idxs_all, i_idxs_all, g_all, fraction=1.0)
a_idxs = a_idxs[:2]
i_idxs = i_idxs[:2]
print(f"Using {len(a_idxs)} excitations")

# LSI-LUSCC
lsi_luscc = LSI_LUSCC(lsi, a_idxs, i_idxs, state=0, threshold=SI_THRESHOLD)
e_roots, si_rq = lsi_luscc.kernel()
print("LSI-LUSCC ground state energy =", e_roots[0])
print("nroots =", lsi_luscc.nroots)
