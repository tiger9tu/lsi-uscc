"""Benchmark las-luscc vs LASSIS for stilbene across all dihedral geometries.

Records for each geometry:
  - LASSIS  : energy, runtime, nroots (size of SI space)
  - las-luscc: energy, runtime (excl. gradient eval), nroots — per fraction
"""

import sys
import numpy as np
from pyscf import gto, scf, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.citools import grad
from mrh.my_pyscf import lassi
from lcc import LSI_LUSCC
from helper.util import get_sorted_excitations
from pathlib import Path
from time import time

# ── Parameters ────────────────────────────────────────────────────────────────
DIHEDRALS = [1, 60, 90, 120, 180]
FRACS     = [0.01, 0.02, 0.03, 0.04]

ncas_f      = (4, 2, 4)
nelecas_f   = (4, 2, 4)
spin_sub_f  = (1, 1, 1)
frag_atom_list = [
    [1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19],
    [0, 7, 14, 20],
    [8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25],
]
basis   = '6-31g'
verbose = 3

pwd      = Path(__file__).resolve().parent
geom_dir = pwd.parent / 'geom'
data_dir = pwd / 'data'
data_dir.mkdir(exist_ok=True)

# ── Main loop ─────────────────────────────────────────────────────────────────
for dihedral in DIHEDRALS:
    geom_name = f'stil{dihedral:03d}' if dihedral == 1 else f'stil{dihedral}'
    geom_path = geom_dir / f'{geom_name}.xyz'
    log_path  = data_dir / f'{geom_name}.log'

    print('\n' + '=' * 70)
    print(f'Geometry: {geom_name}')
    print('=' * 70)

    with geom_path.open('r') as f:
        xyz = f.read()

    # ── Molecule + RHF ────────────────────────────────────────────────────────
    mol = gto.M(atom=xyz, basis=basis, output=str(log_path), verbose=verbose)
    mf  = scf.RHF(mol).run()

    # ── LASSCF ────────────────────────────────────────────────────────────────
    las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f, verbose=verbose)
    mo_loc = las.localize_init_guess(frag_atom_list, mf.mo_coeff)
    las.kernel(mo_loc)
    print(f'LASSCF energy = {las.e_tot}')

    # ── LASSIS ────────────────────────────────────────────────────────────────
    lsi   = lassi.LASSIS(las)
    t0    = time()
    e_lsi, si = lsi.kernel()
    t_lsi = time() - t0
    nroots_lassis = len(e_lsi)
    print(f'LASSIS energy      = {e_lsi[0]:.10f}')
    print(f'LASSIS nroots      = {nroots_lassis}')
    print(f'LASSIS time        = {t_lsi:.2f} s')

    # ── Gradient (outside luscc timing) ───────────────────────────────────────
    _, g_sel, a_idxs_all, i_idxs_all = grad.get_grad_exact(las, epsilon=0.0)
    g_all = np.array(g_sel)[:, 0]

    # ── las-luscc (sweep over fractions) ──────────────────────────────────────
    for frac in FRACS:
        a_idxs, i_idxs, _ = get_sorted_excitations(
            a_idxs_all, i_idxs_all, g_all, fraction=frac)
        print(f'\n  Fraction: {frac} | nexc: {len(a_idxs)}')

        t0 = time()
        luscc      = LSI_LUSCC(las, a_idxs, i_idxs)
        e_roots, _ = luscc.kernel()
        t_luscc    = time() - t0

        print(f'  las-luscc energy = {e_roots[0]:.10f}')
        print(f'  las-luscc nroots = {luscc.nroots}')
        print(f'  las-luscc time   = {t_luscc:.2f} s')

    sys.stdout.flush()

print('\nDone.')
