"""c10 polyene LAS-LUSCC with absolute gradient threshold eps = 0.0057738."""
import sys
import os
import numpy as np
from pyscf import gto, scf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf.chkfile import load_las_, dump_las
from mrh.exploratory.citools import grad
from lcc import LSI_LUSCC
from pathlib import Path
from time import time

EPSILON = 0.0057738

ncas_f      = (2, 2, 2, 2, 2)
nelecas_f   = (2, 2, 2, 2, 2)
spin_sub_f  = (1, 1, 1, 1, 1)
frag_atom_list = [[0, 2], [10, 12], [18, 19], [13, 11], [3, 1]]
basis   = '6-31g'
verbose = 3
max_memory = 50000

pwd = Path(__file__).resolve().parent
geom_path = pwd.parent / 'geom' / 'c10.xyz'
log_path  = pwd / 'data' / 'c10_eps_abs.log'
chk_path  = str(pwd / 'data' / 'c10.chk')
with geom_path.open('r') as f:
    xyz = f.read()

mol = gto.M(atom=xyz, basis=basis, output=log_path, verbose=verbose,
            max_memory=max_memory)
mf = scf.RHF(mol).run()

las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f, verbose=verbose)
las.max_memory = max_memory
las.chkfile = chk_path
mo_loc = las.localize_init_guess(frag_atom_list, mf.mo_coeff)

if os.path.exists(chk_path):
    print(f"Loading LASSCF from checkpoint {chk_path}"); sys.stdout.flush()
    load_las_(las, chk_path)
    print("LASSCF energy =", las.e_tot); sys.stdout.flush()
else:
    t0 = time()
    las.kernel(mo_loc)
    dump_las(las, chk_path)
    print("LASSCF energy =", las.e_tot)
    print("LASSCF time: {:.2f} s".format(time() - t0)); sys.stdout.flush()

# ── Select with absolute threshold ─────────────────────────────────────────
t0 = time()
g_full, g_sel, a_idxs_sel, i_idxs_sel = grad.get_grad_exact(las, epsilon=EPSILON)
print("get_grad_exact time: {:.2f} s".format(time() - t0))
print("Total amplitudes:", len(g_full))
print(f"Selection threshold: epsilon = {EPSILON}")
print(f"Number of selected excitations: {len(a_idxs_sel)}"); sys.stdout.flush()

# g_sel is list of (gradient, original_index) tuples for the selected ones
selected_grads = np.array([gi[0] for gi in g_sel])
selected_idxs  = np.array([gi[1] for gi in g_sel], dtype=int)

# ── LAS-LUSCC ──────────────────────────────────────────────────────────────
t0 = time()
las_luscc = LSI_LUSCC(las, a_idxs_sel, i_idxs_sel)
e_roots, si = las_luscc.kernel()
si_arr = np.asarray(si)
si0 = si_arr[:, 0]
print("LAS-LUSCC time: {:.2f} s".format(time() - t0))
print("LAS-LUSCC ground state energy =", e_roots[0])
print("LAS-LUSCC number of LUSCC basis states:", si_arr.shape[0])
print("LAS-LUSCC number of roots:", len(e_roots)); sys.stdout.flush()

# ── Save ───────────────────────────────────────────────────────────────────
data = pwd / 'data'
np.save(data / 'c10_eps0057738_si0.npy',        si0)
np.save(data / 'c10_eps0057738_e.npy',          np.asarray(e_roots))
np.save(data / 'c10_eps0057738_grads_sel.npy',  selected_grads)
np.save(data / 'c10_eps0057738_origidx_sel.npy', selected_idxs)
# Save a_idxs and i_idxs as object array (variable-length tuples)
np.save(data / 'c10_eps0057738_a_idxs.npy',
        np.array(a_idxs_sel, dtype=object), allow_pickle=True)
np.save(data / 'c10_eps0057738_i_idxs.npy',
        np.array(i_idxs_sel, dtype=object), allow_pickle=True)

top = np.argsort(-np.abs(si0))[:10]
print("Top-10 |coeff| basis indices:", top.tolist())
print("Top-10 coefficients:", si0[top].tolist())
sys.stdout.flush()
