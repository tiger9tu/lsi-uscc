import sys
import os
import numpy as np
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf.chkfile import load_las_, dump_las
from mrh.exploratory.citools import grad
from mrh.my_pyscf import lassi
from lcc import LSI_LUSCC
from helper.util import get_sorted_excitations
from pathlib import Path
from time import time

# ── 1. System parameters ───────────────────────────────────────────────────
FRACS = [0.05, 0.10, 0.15, 0.20]

ncas_f      = (2, 2, 2, 2, 2)
nelecas_f   = (2, 2, 2, 2, 2)
spin_sub_f  = (1, 1, 1, 1, 1)
frag_atom_list = [[0, 2], [10, 12], [18, 19], [13, 11], [3, 1]]
basis   = '6-31g'
verbose = 3
max_memory = 50000  # 50 GB

pwd = Path(__file__).resolve().parent
geom_path = pwd.parent / 'geom' / 'c10.xyz'
log_path  = pwd / 'data' / 'c10.log'
chk_path  = str(pwd / 'data' / 'c10.chk')
with geom_path.open('r') as f:
    xyz = f.read()

# ── 2. Molecule + RHF ──────────────────────────────────────────────────────
mol = gto.M(atom=xyz, basis=basis, output=log_path, verbose=verbose,
            max_memory=max_memory)
mf = scf.RHF(mol).run()

# ── 3. LASSCF ──────────────────────────────────────────────────────────────
las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f, verbose=verbose)
las.max_memory = max_memory
las.chkfile = chk_path
mo_loc = las.localize_init_guess(frag_atom_list, mf.mo_coeff)

if os.path.exists(chk_path):
    print(f"Loading LASSCF from checkpoint {chk_path}"); sys.stdout.flush()
    t0 = time()
    load_las_(las, chk_path)
    print("LASSCF energy =", las.e_tot)
    print("LASSCF time: 0.00 s (from checkpoint)"); sys.stdout.flush()
else:
    t0 = time()
    las.kernel(mo_loc)
    dump_las(las, chk_path)
    print("LASSCF energy =", las.e_tot)
    print("LASSCF time: {:.2f} s".format(time() - t0)); sys.stdout.flush()

# ── 4. LASSIS (reference) ──────────────────────────────────────────────────
lsi = lassi.LASSIS(las)
t0 = time()
e_lsi, _ = lsi.kernel()
print("LASSIS time: {:.2f} s".format(time() - t0))
print("LASSIS energy =", e_lsi[0])
print("LASSIS number of states:", len(e_lsi)); sys.stdout.flush()

# ── 5. Excitation selection ────────────────────────────────────────────────
_, g_sel, a_idxs_all, i_idxs_all = grad.get_grad_exact(las, epsilon=0.0)
g_all = np.array(g_sel)[:, 0]

# ── 6. LSI_LUSCC (sweep over fractions) ────────────────────────────────────
for frac in FRACS:
    a_idxs, i_idxs, _ = get_sorted_excitations(
        a_idxs_all, i_idxs_all, g_all, fraction=frac)
    print(f"\nFraction: {frac} | Number of excitations: {len(a_idxs)}"); sys.stdout.flush()
    t0 = time()
    lsi_luscc = LSI_LUSCC(las, a_idxs, i_idxs)
    e_roots, si = lsi_luscc.kernel()
    si_arr = np.asarray(si)
    si0 = si_arr[:, 0]
    stem = Path(__file__).stem
    np.save(pwd / 'data' / f"{stem}_frac{frac:.2f}_si0.npy", si0)
    np.save(pwd / 'data' / f"{stem}_frac{frac:.2f}_e.npy", np.asarray(e_roots))
    top = np.argsort(-np.abs(si0))[:10]
    print("LSI-LUSCC ground state energy =", e_roots[0])
    print("LSI-LUSCC number of states:", len(e_roots))
    print("Top-10 |coeff| basis indices:", top.tolist())
    print("Top-10 coefficients:", si0[top].tolist())
    print("Time: {:.2f} s".format(time() - t0)); sys.stdout.flush()
