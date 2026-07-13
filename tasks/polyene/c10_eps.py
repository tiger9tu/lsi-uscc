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

# ── 1. System parameters (same as c10.py) ──────────────────────────────────
ncas_f      = (2, 2, 2, 2, 2)
nelecas_f   = (2, 2, 2, 2, 2)
spin_sub_f  = (1, 1, 1, 1, 1)
frag_atom_list = [[0, 2], [10, 12], [18, 19], [13, 11], [3, 1]]
basis   = '6-31g'
verbose = 3
max_memory = 50000

pwd = Path(__file__).resolve().parent
geom_path = pwd.parent / 'geom' / 'c10.xyz'
log_path  = pwd / 'data' / 'c10_eps.log'
chk_path  = str(pwd / 'data' / 'c10.chk')
with geom_path.open('r') as f:
    xyz = f.read()

# ── 2. Molecule + RHF ──────────────────────────────────────────────────────
mol = gto.M(atom=xyz, basis=basis, output=log_path, verbose=verbose,
            max_memory=max_memory)
mf = scf.RHF(mol).run()

# ── 3. LASSCF (reuse checkpoint) ───────────────────────────────────────────
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

# ── 4. Full gradient computation ───────────────────────────────────────────
t0 = time()
g_all, _, _, _ = grad.get_grad_exact(las, epsilon=0.0)
print("get_grad_exact (epsilon=0) time: {:.2f} s".format(time() - t0))
print("Total number of excitation amplitudes:", len(g_all)); sys.stdout.flush()

# ── 5. Build eps array per user's spec ─────────────────────────────────────
sortg = np.sort(np.abs(g_all))
n = len(g_all)
factors = np.arange(0.01, 0.16, 0.01)
x = np.array([int(np.floor(f * n)) for f in factors])
eps = np.zeros(15)
for i in range(15):
    eps[i] = sortg[-x[i]]

print("\nfactor    x[i]      eps[i]")
for i in range(15):
    print(f"{factors[i]:6.2f}  {x[i]:6d}  {eps[i]:.6e}")
sys.stdout.flush()

np.save(pwd / 'data' / 'c10_eps_thresholds.npy', eps)
np.save(pwd / 'data' / 'c10_eps_factors.npy', factors)

# ── 6. Select excitations using eps[2] (factor 0.03 → top ~3%) ─────────────
epsilon_use = eps[2]
print(f"\nUsing eps[2] = {epsilon_use:.6e} (factor = {factors[2]:.2f})")
t0 = time()
_, _, a_idxs_sel, i_idxs_sel = grad.get_grad_exact(las, epsilon=epsilon_use)
print("get_grad_exact (epsilon=eps[2]) time: {:.2f} s".format(time() - t0))
print("Number of selected excitations:", len(a_idxs_sel)); sys.stdout.flush()

# ── 7. LAS-LUSCC (single LAS reference) ────────────────────────────────────
t0 = time()
las_luscc = LSI_LUSCC(las, a_idxs_sel, i_idxs_sel)
e_roots, si = las_luscc.kernel()
si_arr = np.asarray(si)
si0 = si_arr[:, 0]
print("LAS-LUSCC time: {:.2f} s".format(time() - t0))
print("LAS-LUSCC ground state energy =", e_roots[0])
print("LAS-LUSCC number of states:", len(e_roots))

# ── 8. Save LUSCC amplitudes (ground state coefficients in LUSCC basis) ────
np.save(pwd / 'data' / 'c10_eps2_si0.npy', si0)
np.save(pwd / 'data' / 'c10_eps2_e.npy', np.asarray(e_roots))

top = np.argsort(-np.abs(si0))[:10]
print("Top-10 |coeff| basis indices:", top.tolist())
print("Top-10 coefficients:", si0[top].tolist())
sys.stdout.flush()
