import numpy as np
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.citools import grad
from lcc import LSI_LUSCC
from helper.util import get_sorted_excitations
from pathlib import Path
from time import time

# ── 1. System parameters ───────────────────────────────────────────────────
FRAC = 1.0  # use all excitations

ncas_f      = (2, 2, 2, 2)
nelecas_f   = (2, 2, 2, 2)
spin_sub_f  = (1, 1, 1, 1)
frag_atom_list = [[0, 2], [10, 12], [13, 11], [3, 1]]
basis   = '6-31g'
verbose = lib.logger.DEBUG4

pwd = Path(__file__).resolve().parent
geom_path = pwd.parent / 'geom' / 'c8-lsi.xyz'
log_path  = pwd / 'data' / 'c8.log'
with geom_path.open('r') as f:
    xyz = f.read()

# ── 2. Molecule + RHF ──────────────────────────────────────────────────────
mol = gto.M(atom=xyz, basis=basis, output=log_path, verbose=verbose)
mf = scf.RHF(mol).run()

# ── 3. LASSCF ──────────────────────────────────────────────────────────────
las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f, verbose=verbose)
las.max_memory = 12000
mo_loc = las.localize_init_guess(frag_atom_list, mf.mo_coeff)
las.kernel(mo_loc)
print("LASSCF energy =", las.e_tot)

# ── 5. Excitation selection ────────────────────────────────────────────────
t0 = time()
_, g_sel, a_idxs_all, i_idxs_all = grad.get_grad_exact(las, epsilon=0.0)
print("Gradient calculation: {:.2f} s".format(time() - t0))
g_all = np.array(g_sel)[:, 0]
a_idxs, i_idxs, _ = get_sorted_excitations(
    a_idxs_all, i_idxs_all, g_all, fraction=FRAC)
print(f"Number of excitations: {len(a_idxs)}")

# ── 6. LSI_LUSCC ────────────────────────────────────────────────────────────
t0 = time()
lsi_luscc = LSI_LUSCC(las, a_idxs, i_idxs)
e_roots, si = lsi_luscc.kernel()
print("LSI-LUSCC ground state energy =", e_roots[0])
print("Time: {:.2f} s".format(time() - t0))
