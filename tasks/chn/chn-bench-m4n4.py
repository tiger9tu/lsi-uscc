"""Benchmark LSI-LUSCC(m=4, n=4) vs LASSIrq(r=4, q=4) at dr=1.0.

Times each step individually to identify the cost bottleneck.
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
from mrh.my_pyscf.lassi import LASSIrq
from mrh.exploratory.citools.grad import get_grad_exact_lassi
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from lcc import LSI_LUSCC
from helper.util import get_sorted_excitations

DR = 1.0
M  = 4
N  = 4

ncas_f         = (4, 2, 4)
nelecas_f      = ((2, 2), (1, 1), (2, 2))
spin_sub_f     = (1, 1, 1)
frag_atom_list = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]
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

# ── CASCI reference ───────────────────────────────────────────────────────────
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver = csf_solver(mol, smult=1)
mc.kernel(las.mo_coeff)
e_casci = float(mc.e_tot)
pr(f'CASCI energy  = {e_casci:.8f}')
del mc; gc.collect()

pr('')
pr('=' * 60)
pr('LSI-LUSCC(m=%d, n=%d) benchmark' % (M, N))
pr('=' * 60)

# ── Step 1: LASSIS ────────────────────────────────────────────────────────────
pr('\n[Step 1] LASSIS ...')
t0 = time()
lsi = LASSIS(las); lsi.verbose = lib.logger.INFO
lsi.kernel()
timings['lassis'] = time() - t0
e_lassis = float(lsi.e_roots[0])
pr('[Step 1] LASSIS: E=%.8f  nroots=%d  (%.1fs)' % (e_lassis, len(lsi.e_roots), timings['lassis']))

# ── Step 2: |lsi'> sub-LASSI with top_m ──────────────────────────────────────
pr('\n[Step 2] |lsi prime> sub-LASSI (top_m=%d) ...' % M)
t0 = time()
lsi_prime = LSI_LUSCC(lsi, [], [], top_m=M)
lsi_prime.verbose = lib.logger.INFO
e_prime, _ = lsi_prime.kernel()
timings['lsi_prime'] = time() - t0
m_actual = lsi_prime.nroots
pr('[Step 2] |lsi prime>: E=%.8f  m_actual=%d  (%.1fs)' % (float(e_prime[0]), m_actual, timings['lsi_prime']))

# ── Step 3: Gradients on |lsi'> ───────────────────────────────────────────────
pr('\n[Step 3] Gradients on |lsi prime> ...')
t0 = time()
_, g_sel, a_all, i_all = get_grad_exact_lassi(lsi_prime, state=0)
g_all = np.array([g for g, _ in g_sel])
a_sorted, i_sorted, _ = get_sorted_excitations(a_all, i_all, g_all, fraction=1.0)
n_total = len(a_sorted)
timings['grad'] = time() - t0
pr('[Step 3] Gradients: n_total=%d  (%.1fs)' % (n_total, timings['grad']))

# ── Step 4: LSI-LUSCC diagonalization ─────────────────────────────────────────
pr('\n[Step 4] LSI-LUSCC(m=%d, n=%d) diagonalization ...' % (M, N))
a_use = a_sorted[:N]; i_use = i_sorted[:N]
t0 = time()
luscc = LSI_LUSCC(lsi_prime, a_use, i_use, top_m=M)
luscc.verbose = lib.logger.INFO
e_roots, _ = luscc.kernel()
timings['luscc'] = time() - t0
e_luscc = float(e_roots[0])
nroots_luscc = luscc.nroots
pr('[Step 4] LSI-LUSCC: E=%.8f  nroots=%d  (%.1fs)' % (e_luscc, nroots_luscc, timings['luscc']))
del luscc, e_roots; gc.collect()

t_luscc_total = sum(timings[k] for k in ['lassis','lsi_prime','grad','luscc'])

pr('')
pr('=' * 60)
pr('LSI-LUSCC(m=%d,n=%d) summary  —  %d states' % (M, N, nroots_luscc))
pr('  Step 1  LASSIS       : %7.1f s' % timings['lassis'])
pr('  Step 2  |lsi prime>  : %7.1f s' % timings['lsi_prime'])
pr('  Step 3  Gradients    : %7.1f s' % timings['grad'])
pr('  Step 4  LSI-LUSCC    : %7.1f s' % timings['luscc'])
pr('  Total                : %7.1f s' % t_luscc_total)
pr('  E - E_CASCI          : %+.4f mH' % ((e_luscc - e_casci)*1000))
pr('')

pr('=' * 60)
pr('LASSIrq(r=%d, q=%d) reference  —  9717 states  634.3s total' % (M, N))
pr('  (from chn-lassirq.py run)')
pr('  E - E_CASCI : +6.1592 mH')
pr('=' * 60)
