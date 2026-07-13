"""Stilbene at 90 deg dihedral, SINGLET state, LAS-LUSCC with eps = 0.00486175."""
import os
import sys
import numpy as np
from pathlib import Path
from time import time

from pyscf import gto, scf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf.chkfile import load_las_, dump_las
from mrh.exploratory.citools import grad
from lcc import LSI_LUSCC

EPSILON  = 0.00486175
DIHEDRAL = 90

ncas_f      = (4, 2, 4)
nelecas_f   = (4, 2, 4)
spin_sub_f  = (1, 1, 1)         # singlet on each fragment
frag_atom_list = [
    [1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19],
    [0, 7, 14, 20],
    [8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25],
]
basis      = '6-31g'
verbose    = 3
max_memory = 70000

pwd       = Path(__file__).resolve().parent
data_dir  = pwd / 'data'
data_dir.mkdir(exist_ok=True)
geom_path = pwd.parent / 'geom' / f'stil{DIHEDRAL}.xyz'
log_path  = data_dir / f'stil{DIHEDRAL}_singlet_eps.log'
chk_path  = str(data_dir / f'stil{DIHEDRAL}_singlet.chk')

with geom_path.open('r') as f:
    xyz = f.read()

# ── Molecule + RHF ─────────────────────────────────────────────────────────
mol = gto.M(atom=xyz, basis=basis, output=str(log_path), verbose=verbose,
            max_memory=max_memory)
mf = scf.RHF(mol).run()
print(f"RHF energy = {mf.e_tot}"); sys.stdout.flush()

# ── LASSCF (singlet) ───────────────────────────────────────────────────────
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

# ── Select with absolute gradient threshold ────────────────────────────────
t0 = time()
g_full, g_sel, a_idxs_sel, i_idxs_sel = grad.get_grad_exact(las, epsilon=EPSILON)
print(f"get_grad_exact time: {time()-t0:.2f} s")
print(f"Total amplitudes: {len(g_full)}")
print(f"Selection threshold: epsilon = {EPSILON}")
print(f"Number of selected excitations: {len(a_idxs_sel)}"); sys.stdout.flush()

selected_grads = np.array([gi[0] for gi in g_sel])
selected_idxs  = np.array([gi[1] for gi in g_sel], dtype=int)

# ── LAS-LUSCC ──────────────────────────────────────────────────────────────
t0 = time()
las_luscc = LSI_LUSCC(las, a_idxs_sel, i_idxs_sel)
e_roots, si = las_luscc.kernel()
si_arr = np.asarray(si)
si0    = si_arr[:, 0]
print(f"LAS-LUSCC time: {time()-t0:.2f} s")
print(f"LAS-LUSCC ground state energy = {e_roots[0]}")
print(f"LAS-LUSCC number of LUSCC basis states: {si_arr.shape[0]}")
print(f"LAS-LUSCC number of roots: {len(e_roots)}"); sys.stdout.flush()

# ── Save ───────────────────────────────────────────────────────────────────
tag = f'stil{DIHEDRAL}_singlet_eps{EPSILON:.7f}'.replace('.', 'p')
np.save(data_dir / f'{tag}_si0.npy',         si0)
np.save(data_dir / f'{tag}_e.npy',           np.asarray(e_roots))
np.save(data_dir / f'{tag}_grads_sel.npy',   selected_grads)
np.save(data_dir / f'{tag}_origidx_sel.npy', selected_idxs)
np.save(data_dir / f'{tag}_a_idxs.npy',
        np.array(a_idxs_sel, dtype=object), allow_pickle=True)
np.save(data_dir / f'{tag}_i_idxs.npy',
        np.array(i_idxs_sel, dtype=object), allow_pickle=True)

top = np.argsort(-np.abs(si0))[:10]
print(f"Top-10 |coeff| basis indices: {top.tolist()}")
print(f"Top-10 coefficients: {si0[top].tolist()}")
sys.stdout.flush()
