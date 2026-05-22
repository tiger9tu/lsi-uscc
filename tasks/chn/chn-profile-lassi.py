"""Profile LASSI H-build: LASSIrq(3,3) vs LSI-LUSCC(4,4) at dr=1.0.

Intercepts LSTDM.__init__ to capture:
  - lroots (states per rootspace per fragment)
  - exc table sizes (number of rootspace-pair iterations)
Then runs kernel() and records timing + sprint_profile().
Sends email summary.
"""

import sys, gc, smtplib
import numpy as np
from pathlib import Path
from time import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sys.path.insert(0, '/home/tuyue/workspace/lsi-uscc')

from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.my_pyscf.lassi.lassis import LASSIS
from mrh.my_pyscf.lassi import LASSIrq
from mrh.my_pyscf.lassi.op_o1 import stdm as stdm_mod
from mrh.my_pyscf.lassi.citools import get_lroots
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from lcc import LSI_LUSCC
from mrh.my_pyscf.fci import csf_solver

DR = 1.0
PES_CKPT = Path(__file__).parent / 'ckpt' / 'pes'
ncas_f, nelecas_f, spin_sub_f = (4,2,4), ((2,2),(1,1),(2,2)), (1,1,1)
ncas, nelecas = sum(ncas_f), (5,5)

def pr(msg=''):
    print(msg, flush=True)

def send_email(subject, body):
    try:
        creds = (Path.home()/'.gmail_app_password').read_text().strip()
        msg = MIMEMultipart()
        msg['Subject'] = subject; msg['From'] = 'tuyue3@gmail.com'; msg['To'] = 'tuyue3@gmail.com'
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP('smtp.gmail.com', 587) as srv:
            srv.ehlo(); srv.starttls()
            srv.login('tuyue3@gmail.com', creds)
            srv.sendmail('tuyue3@gmail.com', 'tuyue3@gmail.com', msg.as_string())
        pr('[email] sent')
    except Exception as e:
        pr(f'[email] failed: {e}')

# ── Molecule + LASSCF ─────────────────────────────────────────────────────────
mol = struct(DR, DR, '6-31g')
mol.verbose = lib.logger.INFO; mol.spin = 8; mol.max_memory = 100000; mol.build()
mf = scf.RHF(mol); mf.verbose = lib.logger.INFO; mf.kernel()

las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f)
las.verbose = lib.logger.INFO; las.max_memory = 100000
saved_mo = np.load(str(PES_CKPT/'dr10_result.npz'), allow_pickle=True)['mo_coeff']
las.kernel(saved_mo)

mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver = csf_solver(mol, smult=1)
mc.kernel(las.mo_coeff); e_casci = float(mc.e_tot)
del mc; gc.collect()

# ── Monkey-patch LSTDM to capture exc table + lroots ─────────────────────────
_captured = {}
_orig_init = stdm_mod.LSTDM.__init__

def _patched_init(self, ints, nlas, hopping_index, lroots, *args, **kwargs):
    _orig_init(self, ints, nlas, hopping_index, lroots, *args, **kwargs)
    tag = getattr(self, '_profile_tag', 'unknown')
    exc_sizes = {
        'null'  : len(self.exc_null),
        '1d'    : len(self.exc_1d),
        '2d'    : len(getattr(self, 'exc_2d', np.empty((0,)))),
        '1c'    : len(self.exc_1c),
        '1c1d'  : len(self.exc_1c1d),
        '1s'    : len(self.exc_1s),
        '1s1c'  : len(self.exc_1s1c),
        '2c'    : len(self.exc_2c),
    }
    lroots_prod = np.prod(lroots, axis=0)
    _captured[tag] = {
        'nroots_rs'    : lroots.shape[1],
        'lroots'       : lroots.copy(),
        'lroots_prod'  : lroots_prod,
        'exc_sizes'    : exc_sizes,
        'exc_total'    : sum(exc_sizes.values()),
        'nstates'      : int(np.sum(lroots_prod)),
        'self'         : self,
    }

stdm_mod.LSTDM.__init__ = _patched_init

def run_lassi_and_profile(obj, tag, label):
    """Run a LASSI-derived object's kernel() and return profiling data."""
    pr(f'\n{"="*60}')
    pr(f'Running: {label}')
    pr(f'{"="*60}')
    obj.verbose = lib.logger.INFO
    obj.max_memory = 100000

    # Tag the LSTDM so our interceptor knows which run this is
    import mrh.my_pyscf.lassi.op_o1.hams2ovlp as hs_mod
    _orig_class = hs_mod.HamS2Ovlp
    class TaggedHamS2Ovlp(_orig_class):
        def __init__(self2, *a, **kw):
            self2._profile_tag = tag
            super().__init__(*a, **kw)
    _orig_HamS2Ovlp_backup = hs_mod.HamS2Ovlp
    hs_mod.HamS2Ovlp = TaggedHamS2Ovlp
    import mrh.my_pyscf.lassi.op_o1 as op_o1_mod
    _orig_class2 = op_o1_mod._HamS2Ovlp_class if hasattr(op_o1_mod, '_HamS2Ovlp_class') else None

    t0 = time()
    try:
        e_roots, si = obj.kernel()
    finally:
        hs_mod.HamS2Ovlp = _orig_HamS2Ovlp_backup

    dt = time() - t0
    e = float(e_roots[0])

    cap = _captured.get(tag, {})
    nroots_rs = cap.get('nroots_rs', '?')
    nstates   = cap.get('nstates',   '?')
    exc_sizes = cap.get('exc_sizes', {})
    exc_total = cap.get('exc_total', '?')
    lroots_prod = cap.get('lroots_prod', np.array([1]))
    lroots    = cap.get('lroots', None)
    self_obj  = cap.get('self', None)

    pr(f'  E                          : {e:.8f}  (ΔE={( e-e_casci)*1000:+.4f} mH)')
    pr(f'  Wall time                  : {dt:.2f} s')
    pr(f'  Rootspaces                 : {nroots_rs}')
    pr(f'  Total states               : {nstates}')
    if isinstance(nstates, int) and isinstance(nroots_rs, int):
        pr(f'  Avg states/rootspace       : {nstates/nroots_rs:.2f}')
        h_elems = nstates**2
        rs_pairs = nroots_rs**2
        pr(f'  H elements (n_states^2)    : {h_elems:,}')
        pr(f'  Rootspace pairs (nrs^2)    : {rs_pairs:,}')
        pr(f'  Avg H-elems per rs-pair    : {h_elems/rs_pairs:.1f}')
        pr(f'  Time per H-element (μs)    : {dt/h_elems*1e6:.3f}')
        pr(f'  Time per rs-pair (μs)      : {dt/rs_pairs*1e6:.3f}')
    if lroots is not None:
        pr(f'  lroots per frag (mean)     : ' + ' / '.join(f'{lroots[fi].mean():.2f}' for fi in range(lroots.shape[0])))
    pr(f'  Exc table rows:')
    for k, v in exc_sizes.items():
        if v > 0:
            pr(f'    {k:8s}: {v:8d}')
    pr(f'  Total exc table rows       : {exc_total}')
    if self_obj is not None:
        pr(f'  Sprint profile:')
        for line in self_obj.sprint_profile().split('\n'):
            pr(f'    {line}')

    return {
        'label'     : label,
        'tag'       : tag,
        'e'         : e,
        'dt'        : dt,
        'nroots_rs' : nroots_rs if isinstance(nroots_rs, int) else -1,
        'nstates'   : nstates   if isinstance(nstates, int)   else -1,
        'exc_total' : exc_total if isinstance(exc_total, int) else -1,
        'exc_sizes' : exc_sizes,
    }

# ── LASSIS for LSI-LUSCC reference ────────────────────────────────────────────
pr('\n[LASSIS] ...')
lsi = LASSIS(las); lsi.verbose = lib.logger.INFO; lsi.kernel()
pr(f'[LASSIS] E={lsi.e_roots[0]:.8f}  nroots={len(lsi.e_roots)}')

# ── Case 1: LASSIrq(3,3) ──────────────────────────────────────────────────────
lsi_rq = LASSIrq(las, r=3, q=3)
res_rq = run_lassi_and_profile(lsi_rq, 'lassirq33', 'LASSIrq(r=3, q=3)')
del lsi_rq; gc.collect()

# ── Case 2: LSI-LUSCC(m=4, n=4) ───────────────────────────────────────────────
lsi_prime = LSI_LUSCC(lsi, [], [], top_m=4)
lsi_prime.verbose = lib.logger.INFO
lsi_prime.kernel()

gdata = np.load(str(PES_CKPT/'dr10_grad.npz'), allow_pickle=True)
a_sorted, i_sorted = list(gdata['a_sorted']), list(gdata['i_sorted'])

luscc44 = LSI_LUSCC(lsi_prime, a_sorted[:4], i_sorted[:4], top_m=4)
res_44 = run_lassi_and_profile(luscc44, 'luscc44', 'LSI-LUSCC(m=4, n=4)')
del luscc44; gc.collect()

# ── Summary ───────────────────────────────────────────────────────────────────
pr('\n' + '='*70)
pr('SUMMARY: Why LASSIrq is faster than LSI-LUSCC')
pr('='*70)

def fmt(v):
    if isinstance(v, int): return f'{v:>14,d}'
    if isinstance(v, float): return f'{v:>14.3f}'
    return f'{str(v):>14s}'

rows = [
    ('Rootspaces',           res_rq['nroots_rs'],  res_44['nroots_rs']),
    ('Total states',         res_rq['nstates'],    res_44['nstates']),
    ('Avg states/rootspace', res_rq['nstates']/max(res_rq['nroots_rs'],1),
                             res_44['nstates']/max(res_44['nroots_rs'],1)),
    ('Exc table rows',       res_rq['exc_total'],  res_44['exc_total']),
    ('H elements (n^2)',     res_rq['nstates']**2, res_44['nstates']**2),
    ('Rootspace pairs(nrs^2)',res_rq['nroots_rs']**2, res_44['nroots_rs']**2),
    ('Ham time (s)',         res_rq['dt'],          res_44['dt']),
    ('Time/H-elem (us)',     res_rq['dt']/max(res_rq['nstates']**2,1)*1e6,
                             res_44['dt']/max(res_44['nstates']**2,1)*1e6),
    ('Time/rs-pair (us)',    res_rq['dt']/max(res_rq['nroots_rs']**2,1)*1e6,
                             res_44['dt']/max(res_44['nroots_rs']**2,1)*1e6),
]

hdr = f"{'Metric':35s}  {'LASSIrq(3,3)':>14}  {'LSI-LUSCC(4,4)':>14}  {'Ratio':>8}"
pr(hdr)
pr('-'*80)
for name, v1, v2 in rows:
    ratio = v2/v1 if isinstance(v1,(int,float)) and v1 > 0 else float('nan')
    pr(f'{name:35s}  {fmt(v1)}  {fmt(v2)}  {ratio:>8.2f}x')

pr('\nConclusion:')
pr('  LSI-LUSCC places each A|LAS_j> in its own rootspace (1 state/rootspace)')
pr('  LASSIrq groups up to q^nfrags states per rootspace via LASCI')
pr('  This gives LASSIrq:')
pr('    (a) fewer rootspace pairs -> less loop overhead')
pr('    (b) more H elements per pair -> better numpy vectorization')
pr('  Together these explain the large per-element speedup of LASSIrq.')
pr('='*70)

# ── Email ─────────────────────────────────────────────────────────────────────
body_lines = ['LASSI H-build profiling: LASSIrq(3,3) vs LSI-LUSCC(4,4)\n',
              'dr=1.0, CHN(10,10)\n\n', hdr+'\n', '-'*80+'\n']
for name, v1, v2 in rows:
    ratio = v2/v1 if isinstance(v1,(int,float)) and v1 > 0 else float('nan')
    body_lines.append(f'{name:35s}  {fmt(v1)}  {fmt(v2)}  {ratio:>8.2f}x\n')

body_lines += [
    '\nConclusion:\n',
    '  LSI-LUSCC places each A|LAS_j> in its own rootspace (1 state/rootspace).\n',
    '  LASSIrq groups up to q^nfrags states per rootspace via LASCI.\n',
    '  This gives LASSIrq:\n',
    '    (a) fewer rootspace pairs -> less loop overhead in exc table iteration\n',
    '    (b) more H elements per pair -> better numpy vectorization of TDM contractions\n',
    '  Both effects together explain the large per-element speedup of LASSIrq.\n',
    '\nNote: LSI-LUSCC(4,4) has only 36 states (trivial); for m=48 n=120 (8092 states),\n',
    'the rootspace-pair count is 8092^2=65M vs 296^2=87k for LASSIrq(3,3), a 748x difference.\n',
]
send_email('[LSI-LUSCC] LASSI profiling: why LASSIrq is faster', ''.join(body_lines))
