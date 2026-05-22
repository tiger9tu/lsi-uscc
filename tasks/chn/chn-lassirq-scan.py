"""LASSIrq large-q scan: r=1..4, q=9,10,11,... until chemical accuracy or timeout.

Stops when:
  - r=4 energy is within chemical accuracy (ΔE < 1.594 mH of CASCI), OR
  - r=4 runtime exceeds MAX_DT_R4 seconds for a single run.

Checkpoints every (r,q) pair; safe to interrupt and resume.
"""

import gc, sys, smtplib
import numpy as np
from pathlib import Path
from time import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sys.path.insert(0, '/home/tuyue/workspace/lsi-uscc')

from pyscf import scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSIrq
from mrh.tests.lasscf.c2h4n4_struct import structure as struct

DR          = 1.0
RS          = [1, 2, 3, 4]
Q_START     = 9
Q_MAX       = 30
CHEM_ACC_MH = 1.594          # 1 kcal/mol in mH
MAX_DT_R4   = 7200.0         # 2 hours

ncas_f     = (4, 2, 4)
nelecas_f  = ((2, 2), (1, 1), (2, 2))
spin_sub_f = (1, 1, 1)
ncas, nelecas = sum(ncas_f), (5, 5)

CKPT_DIR = Path(__file__).parent / 'ckpt' / 'lassirq'
CKPT_DIR.mkdir(parents=True, exist_ok=True)
PES_CKPT = Path(__file__).parent / 'ckpt' / 'pes'


def pr(msg=''):
    print(msg, flush=True)


def send_email(subject, body):
    try:
        creds = (Path.home() / '.gmail_app_password').read_text().strip()
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From']    = 'tuyue3@gmail.com'
        msg['To']      = 'tuyue3@gmail.com'
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP('smtp.gmail.com', 587) as srv:
            srv.ehlo(); srv.starttls()
            srv.login('tuyue3@gmail.com', creds)
            srv.sendmail('tuyue3@gmail.com', 'tuyue3@gmail.com', msg.as_string())
        pr('[email] sent')
    except Exception as e:
        pr(f'[email] failed: {e}')


pr('=' * 70)
pr(f'LASSIrq large-q scan  dr={DR}  r={RS}  q={Q_START}..{Q_MAX}')
pr(f'Stop when: ΔE(r=4) < {CHEM_ACC_MH} mH  OR  dt(r=4) > {MAX_DT_R4:.0f} s')
pr('=' * 70)

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
e_lasscf = float(las.e_states[0])

mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver = csf_solver(mol, smult=1)
mc.verbose    = lib.logger.INFO
mc.kernel(las.mo_coeff)
e_casci = float(mc.e_tot)
del mc; gc.collect()

pr(f'LASSCF = {e_lasscf:.8f}  CASCI = {e_casci:.8f}')

# ── Scan ──────────────────────────────────────────────────────────────────────
results  = {}   # (r,q) -> (e, nroots, dt)
stop_reason = None

for q in range(Q_START, Q_MAX + 1):
    pr(f'\n{"─"*60}')
    pr(f'q = {q}')
    pr(f'{"─"*60}')

    for r in RS:
        tag  = f'r{r}_q{q}'
        ckpt = CKPT_DIR / f'result_dr10_{tag}.npz'

        if ckpt.exists():
            d      = np.load(str(ckpt), allow_pickle=True)
            e      = float(d['e'])
            nroots = int(d['nroots'])
            dt     = float(d['dt'])
            pr(f'[{tag}] loaded from ckpt: E={e:.8f}  nroots={nroots}  ({dt:.1f}s)')
        else:
            pr(f'[{tag}] Running LASSIrq(r={r}, q={q}) ...')
            lsi = LASSIrq(las, r=r, q=q)
            lsi.verbose    = lib.logger.INFO
            lsi.max_memory = 100000
            t0 = time()
            try:
                e_roots, si = lsi.kernel()
                dt     = time() - t0
                e      = float(e_roots[0])
                nroots = int(si.shape[1]) if si is not None else len(e_roots)
                delta  = (e - e_casci) * 1000
                pr(f'[{tag}] E={e:.8f}  ΔE={delta:+.4f} mH  nroots={nroots}  ({dt:.1f}s)')
                np.savez(str(ckpt), e=np.array(e), nroots=np.array(nroots), dt=np.array(dt))
                pr(f'[{tag}] checkpoint saved')
            except Exception as exc:
                import traceback
                dt = time() - t0
                pr(f'[{tag}] ERROR after {dt:.1f}s: {exc}')
                traceback.print_exc(file=sys.stdout)
                results[(r, q)] = (None, None, dt)
                continue
            finally:
                try: del lsi, e_roots, si
                except NameError: pass
                gc.collect()

        results[(r, q)] = (e, nroots, dt)

    # ── Check stopping criteria after each q level ────────────────────────────
    r4_res = results.get((4, q))
    if r4_res and r4_res[0] is not None:
        delta_r4 = (r4_res[0] - e_casci) * 1000
        dt_r4    = r4_res[2]
        pr(f'\n  r=4,q={q}: ΔE={delta_r4:+.4f} mH  dt={dt_r4:.0f}s')
        if delta_r4 < CHEM_ACC_MH:
            stop_reason = f'chemical accuracy reached at q={q}: ΔE={delta_r4:+.4f} mH'
            pr(f'  *** {stop_reason} ***')
            break
        if dt_r4 > MAX_DT_R4:
            stop_reason = f'timeout at q={q}: r=4 took {dt_r4:.0f}s > {MAX_DT_R4:.0f}s limit'
            pr(f'  *** {stop_reason} ***')
            break
    else:
        pr(f'  r=4,q={q}: no result, stopping')
        stop_reason = f'r=4 failed at q={q}'
        break

# ── Summary ───────────────────────────────────────────────────────────────────
qs_done = sorted(set(q for (r, q) in results))

pr('')
pr('=' * 70)
pr(f'LASSIrq large-q scan  —  E − E_CASCI [mH]')
pr(f'Stop reason: {stop_reason}')
pr(f'CASCI = {e_casci:.8f} Eh')
pr('=' * 70)

q_labels = '  '.join(f'{"q="+str(q):>10}' for q in qs_done)
rq_label = 'r\\q'
header = f"{rq_label:>5}  {q_labels}"
pr(header)
pr('-' * len(header))
table_rows = []
for r in RS:
    cells = []
    for q in qs_done:
        res = results.get((r, q), (None, None, None))
        if res[0] is not None:
            cells.append(f'{(res[0]-e_casci)*1000:+10.4f}')
        else:
            cells.append(f"{'---':>10}")
    row = f"{'r='+str(r):>5}  " + '  '.join(cells)
    pr(row)
    table_rows.append(row)
pr('=' * 70)

pr('\nnroots:')
for r in RS:
    cells = []
    for q in qs_done:
        res = results.get((r, q), (None, None, None))
        cells.append(f'{res[1]:>10d}' if res[1] is not None else f"{'---':>10}")
    pr(f"{'r='+str(r):>5}  " + '  '.join(cells))

pr('\nwall time (s):')
for r in RS:
    cells = []
    for q in qs_done:
        res = results.get((r, q), (None, None, None))
        cells.append(f'{res[2]:>10.0f}' if res[2] is not None else f"{'---':>10}")
    pr(f"{'r='+str(r):>5}  " + '  '.join(cells))

# ── Email ─────────────────────────────────────────────────────────────────────
body_lines = [
    f'LASSIrq CHN(10,10) large-q scan complete.\n',
    f'Stop reason: {stop_reason}\n',
    f'CASCI = {e_casci:.8f} Eh\n\n',
    'E − E_CASCI [mH]:\n',
    header + '\n', '-' * len(header) + '\n',
]
for row in table_rows:
    body_lines.append(row + '\n')
body_lines.append('\nnroots:\n')
for r in RS:
    cells = [f'{results.get((r,q),(None,None,None))[1]:>10d}'
             if results.get((r,q),(None,None,None))[1] is not None else f"{'---':>10}"
             for q in qs_done]
    body_lines.append(f"{'r='+str(r):>5}  " + '  '.join(cells) + '\n')
body_lines.append('\nwall time (s):\n')
for r in RS:
    cells = [f'{results.get((r,q),(None,None,None))[2]:>10.0f}'
             if results.get((r,q),(None,None,None))[2] is not None else f"{'---':>10}"
             for q in qs_done]
    body_lines.append(f"{'r='+str(r):>5}  " + '  '.join(cells) + '\n')

send_email('[LSI-LUSCC] LASSIrq large-q scan complete', ''.join(body_lines))
