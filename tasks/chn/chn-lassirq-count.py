"""Count H matrix elements evaluated by LASSIrq(r=3, q=3) at dr=1.0.

Monkey-patches iterate_subspace_blocks to record each symmetry block size.
Total H elements = sum(n_b^2) over all blocks.
"""

import sys, gc
import numpy as np
from pathlib import Path
from time import time

sys.path.insert(0, '/home/tuyue/workspace/lsi-uscc')

from pyscf import gto, scf, lib
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSIrq
import mrh.my_pyscf.lassi.lassi as _lassi_mod
from mrh.tests.lasscf.c2h4n4_struct import structure as struct

DR, R, Q = 1.0, 3, 3

ncas_f     = (4, 2, 4)
nelecas_f  = ((2, 2), (1, 1), (2, 2))
spin_sub_f = (1, 1, 1)
PES_CKPT   = Path(__file__).parent / 'ckpt' / 'pes'

def pr(msg=''):
    print(msg, flush=True)

# ── Molecule + LASSCF ─────────────────────────────────────────────────────────
mol = struct(DR, DR, '6-31g')
mol.verbose    = lib.logger.INFO
mol.spin       = 8
mol.max_memory = 100000
mol.build()
mf = scf.RHF(mol); mf.verbose = lib.logger.INFO; mf.kernel()

las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f)
las.verbose    = lib.logger.INFO
las.max_memory = 100000
saved_mo = np.load(str(PES_CKPT / 'dr10_result.npz'), allow_pickle=True)['mo_coeff']
las.kernel(saved_mo)
pr(f'LASSCF converged: E={las.e_states[0]:.8f}')

# ── LASSIrq with block-size instrumentation ───────────────────────────────────
pr(f'\nRunning LASSIrq(r={R}, q={Q}) with block-size counting ...')

block_sizes = []
_orig_isb = _lassi_mod.iterate_subspace_blocks

def _patched_isb(las_obj, ci, spacesym, *args, **kwargs):
    for las1, sym, indices, indexed in _orig_isb(las_obj, ci, spacesym, *args, **kwargs):
        n_in_block = int(np.sum(indices[1]))
        block_sizes.append(n_in_block)
        yield las1, sym, indices, indexed

_lassi_mod.iterate_subspace_blocks = _patched_isb
try:
    lsi = LASSIrq(las, r=R, q=Q)
    lsi.verbose = lib.logger.INFO
    t0 = time()
    e_roots, si = lsi.kernel()
    dt = time() - t0
finally:
    _lassi_mod.iterate_subspace_blocks = _orig_isb

e = float(e_roots[0])
nroots = si.shape[1]

# ── Report ────────────────────────────────────────────────────────────────────
block_sizes = np.array(block_sizes)
total_elements = int(np.sum(block_sizes**2))
n_blocks       = len(block_sizes)

pr('')
pr('=' * 60)
pr(f'LASSIrq(r={R}, q={Q})  dr={DR}')
pr(f'  Total states (nroots)     : {nroots}')
pr(f'  Wall time                 : {dt:.1f} s')
pr(f'  E                         : {e:.8f}')
pr('')
pr(f'  Symmetry blocks           : {n_blocks}')
pr(f'  Block sizes (min/max/mean): {block_sizes.min()} / {block_sizes.max()} / {block_sizes.mean():.1f}')
pr(f'  Block size distribution:')
unique, counts = np.unique(block_sizes, return_counts=True)
for u, c in zip(unique, counts):
    pr(f'    size={u:4d}  count={c:4d}  H-elements={u*u:8d}  subtotal={u*u*c:10d}')
pr('')
pr(f'  Total H matrix elements   : {total_elements:,}')
pr(f'  (= sum of n_b^2 over all blocks)')
pr('')
pr(f'  Compare: LSI-LUSCC(m=48,n=120) — 1 block of 8092 states')
pr(f'           H elements = 8092^2 = {8092**2:,}')
pr(f'           Ratio: {8092**2 / total_elements:.1f}x more')
pr('=' * 60)
