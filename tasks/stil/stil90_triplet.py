"""Stilbene at 90° dihedral, lowest TRIPLET state.

Triplet placement: central C=C fragment (frag 1) — at 90° dihedral, this is the
diradical site, so the lowest triplet has its two unpaired electrons there.
Phenyl fragments stay singlet. Global S = 1.

Excitation selection: all single+double excitations with |gradient| >= 0.00252
(absolute threshold, not a fraction).
"""

import os
import sys
import numpy as np
from pathlib import Path
from time import time

from pyscf import gto, scf, lib
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf.chkfile import load_las_, dump_las
from mrh.exploratory.citools import grad
from mrh.my_pyscf import lassi

from lcc import LSI_LUSCC
from helper.util import get_sorted_excitations

# ── 1. System parameters ───────────────────────────────────────────────────
DIHEDRAL = 90

# Triplet on central C=C; singlets on phenyls -> global S = 1
ncas_f      = (4, 2, 4)
nelecas_f   = (4, 2, 4)
spin_sub_f  = (1, 3, 1)
frag_atom_list = [
    [1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19],
    [0, 7, 14, 20],
    [8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25],
]
basis      = '6-31g'
verbose    = 3
max_memory = 70000  # MB

# Gradient threshold: |g| >= 0.00252 (helper uses strict '>', subtract tiny eps)
GRAD_THRESH = 0.00252
EPS_FOR_GT  = GRAD_THRESH - 1e-12

pwd       = Path(__file__).resolve().parent
data_dir  = pwd / 'data'
data_dir.mkdir(exist_ok=True)
geom_path = pwd.parent / 'geom' / f'stil{DIHEDRAL}.xyz'
log_path  = data_dir / f'stil{DIHEDRAL}_triplet.log'
chk_path  = str(data_dir / f'stil{DIHEDRAL}_triplet.chk')

with geom_path.open('r') as f:
    xyz = f.read()

# ── 2. Molecule + ROHF ─────────────────────────────────────────────────────
mol = gto.M(atom=xyz, basis=basis, spin=2,
            output=str(log_path), verbose=verbose,
            max_memory=max_memory)
mf = scf.ROHF(mol).run()
print(f"ROHF energy = {mf.e_tot}"); sys.stdout.flush()

# ── 3. LASSCF (triplet) ────────────────────────────────────────────────────
las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f, verbose=verbose)
las.max_memory = max_memory
las.chkfile    = chk_path
mo_loc = las.localize_init_guess(frag_atom_list, mf.mo_coeff)

if os.path.exists(chk_path):
    print(f"Loading LASSCF from checkpoint {chk_path}"); sys.stdout.flush()
    load_las_(las, chk_path)
    print(f"LASSCF energy = {las.e_tot}"); sys.stdout.flush()
else:
    t0 = time()
    las.kernel(mo_loc)
    dump_las(las, chk_path)
    print(f"LASSCF energy = {las.e_tot}")
    print(f"LASSCF time: {time()-t0:.2f} s"); sys.stdout.flush()

# ── 4. LASSIS (triplet reference) ──────────────────────────────────────────
lsi = lassi.LASSIS(las)
t0 = time()
e_lsi, _ = lsi.kernel()
print(f"LASSIS time: {time()-t0:.2f} s")
print(f"LASSIS energy (root 0) = {e_lsi[0]}")
print(f"LASSIS number of states: {len(e_lsi)}"); sys.stdout.flush()

# ── 5. Excitation selection (absolute gradient threshold) ──────────────────
_, g_sel, a_idxs_all, i_idxs_all = grad.get_grad_exact(las, epsilon=0.0)
g_all = np.array(g_sel)[:, 0]

# Quick gradient distribution for diagnostics
abs_g = np.abs(g_all)
for thr in (1e-5, 1e-4, 1e-3, 2.52e-3, 5e-3, 1e-2):
    print(f"  #|g| >= {thr:.5g} : {int((abs_g >= thr).sum())}")
sys.stdout.flush()

a_idxs, i_idxs, _ = get_sorted_excitations(
    a_idxs_all, i_idxs_all, g_all, epsilon=EPS_FOR_GT)
print(f"\nGradient threshold: |g| >= {GRAD_THRESH} | Number of excitations: {len(a_idxs)}")
sys.stdout.flush()

# ── 6. LSI_LUSCC ───────────────────────────────────────────────────────────
t0 = time()
lsi_luscc = LSI_LUSCC(las, a_idxs, i_idxs)
e_roots, si = lsi_luscc.kernel()

si_arr = np.asarray(si)
si0    = si_arr[:, 0]

np.save(data_dir / f"stil{DIHEDRAL}_triplet_si0.npy", si0)
np.save(data_dir / f"stil{DIHEDRAL}_triplet_e.npy",   np.asarray(e_roots))

top = np.argsort(-np.abs(si0))[:10]
print(f"LSI-LUSCC ground state energy = {e_roots[0]}")
print(f"LSI-LUSCC number of roots: {len(e_roots)}")
print(f"LSI-LUSCC basis dim: {len(si0)}")
print(f"Top-10 |coeff| basis indices: {top.tolist()}")
print(f"Top-10 coefficients: {si0[top].tolist()}")
print(f"Time: {time()-t0:.2f} s"); sys.stdout.flush()
