"""Benchmark LSI-LUSCC(m=48, n=120, opt=1) at dr=1.0.

Times each step to compare fairly with LASSIrq(r=4,q=4) = 634s / 9717 states.
Gradients loaded from PES checkpoint (already computed on |lsi'>(m=48)).
"""

import sys, gc
import numpy as np
from pathlib import Path
from time import time

sys.path.insert(0, '/home/tuyue/workspace/lsi-uscc')

from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.my_pyscf.lassi.lassis import LASSIS
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from lcc import LSI_LUSCC

DR = 1.0
M  = 48
N  = 120

ncas_f         = (4, 2, 4)
nelecas_f      = ((2, 2), (1, 1), (2, 2))
spin_sub_f     = (1, 1, 1)
ncas, nelecas  = sum(ncas_f), (5, 5)
PES_CKPT = Path(__file__).parent / 'ckpt' / 'pes'

def pr(msg=''):
    print(msg, flush=True)

timings = {}

# ── Molecule + RHF + LASSCF ───────────────────────────────────────────────────
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
pr(f'LASSCF energy = {las.e_states[0]:.8f}')

mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver = csf_solver(mol, smult=1)
mc.kernel(las.mo_coeff)
e_casci = float(mc.e_tot)
pr(f'CASCI energy  = {e_casci:.8f}')
del mc; gc.collect()

pr('')
pr('=' * 60)
pr('LSI-LUSCC(m=%d, n=%d, opt=1) benchmark' % (M, N))
pr('=' * 60)

# ── Step 1: LASSIS ────────────────────────────────────────────────────────────
pr('\n[Step 1] LASSIS ...')
t0 = time()
lsi = LASSIS(las); lsi.verbose = lib.logger.INFO
lsi.kernel()
timings['lassis'] = time() - t0
pr('[Step 1] LASSIS: E=%.8f  nroots=%d  (%.1fs)' % (
    float(lsi.e_roots[0]), len(lsi.e_roots), timings['lassis']))

# ── Step 2: |lsi'> sub-LASSI ─────────────────────────────────────────────────
pr('\n[Step 2] |lsi prime> (top_m=%d) ...' % M)
t0 = time()
lsi_prime = LSI_LUSCC(lsi, [], [], top_m=M)
lsi_prime.verbose = lib.logger.INFO
e_prime, _ = lsi_prime.kernel()
timings['lsi_prime'] = time() - t0
m_actual = lsi_prime.nroots
pr('[Step 2] |lsi prime>: E=%.8f  m_actual=%d  (%.1fs)' % (
    float(e_prime[0]), m_actual, timings['lsi_prime']))

# ── Step 3: Load gradients from PES checkpoint ────────────────────────────────
pr('\n[Step 3] Loading gradients from PES checkpoint ...')
gdata = np.load(str(PES_CKPT / 'dr10_grad.npz'), allow_pickle=True)
a_sorted = list(gdata['a_sorted'])
i_sorted = list(gdata['i_sorted'])
n_total  = int(gdata['n_total'])
t_grad   = float(gdata['t_grad'])
pr('[Step 3] Loaded %d excitations (originally computed in %.1fs)' % (n_total, t_grad))
a_use = a_sorted[:N]; i_use = i_sorted[:N]
pr('[Step 3] Using top %d excitations' % N)
timings['grad'] = 0.0  # loaded from ckpt

# ── Step 4: LSI-LUSCC diagonalization (opt=1) ─────────────────────────────────
pr('\n[Step 4] LSI-LUSCC(m=%d, n=%d, opt=1) ...' % (M, N))
luscc = LSI_LUSCC(lsi_prime, a_use, i_use, top_m=M, opt=1)
luscc.verbose = lib.logger.INFO
t0 = time()
e_roots, _ = luscc.kernel()
timings['luscc'] = time() - t0
e_luscc   = float(e_roots[0])
nroots    = luscc.nroots
pr('[Step 4] E=%.8f  nroots=%d  (%.1fs)' % (e_luscc, nroots, timings['luscc']))
del luscc, e_roots; gc.collect()

pr('')
pr('=' * 60)
pr('LSI-LUSCC(m=%d, n=%d, opt=1)  —  %d states' % (M, N, nroots))
pr('  Step 1  LASSIS           : %7.1f s' % timings['lassis'])
pr('  Step 2  |lsi prime>      : %7.1f s' % timings['lsi_prime'])
pr('  Step 3  Gradients (ckpt) : %7.1f s  [orig: %.1fs]' % (0.0, t_grad))
pr('  Step 4  Diagonalize opt=1: %7.1f s' % timings['luscc'])
pr('  Total (excl. grad)       : %7.1f s' % sum(timings.values()))
pr('  E - E_CASCI              : %+.4f mH' % ((e_luscc - e_casci)*1000))
pr('')
pr('  Compare: LASSIrq(r=4,q=4) — 9717 states — 634s — +6.16 mH')
pr('           TrackedLSI-LUSCC opt=0 (heatmap) — 8092 states — 9933s')
pr('=' * 60)
