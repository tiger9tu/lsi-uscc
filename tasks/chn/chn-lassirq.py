"""LASSIrq for CHN(10,10) at dr=1.0, r=1..5, q=1..8 (40 combinations).

Setup mirrors chn-pes.py:
  - LASSCF(mf, (4,2,4), ((2,2),(1,1),(2,2)), spin_sub=(1,1,1))
  - mol.spin=8, mol.max_memory=100000
  - Start from saved mo_coeff at ckpt/pes/dr10_result.npz

Run from /home/tuyue/workspace/lsi-uscc/:
  source /home/tuyue/workspace/venv/bin/activate
  python -u tasks/chn/chn-lassirq.py 2>&1 | tee tasks/chn/chn-lassirq8.log
"""

import gc
import sys
import numpy as np
import smtplib
from pathlib import Path
from time import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sys.path.insert(0, '/home/tuyue/workspace/lsi-uscc')

from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSIrq
from mrh.tests.lasscf.c2h4n4_struct import structure as struct

# ── Parameters ────────────────────────────────────────────────────────────────
DR   = 1.0
RS   = [1, 2, 3, 4, 5]
QS   = [1, 2, 3, 4, 5, 6, 7, 8]

ncas_f         = (4, 2, 4)
nelecas_f      = ((2, 2), (1, 1), (2, 2))
spin_sub_f     = (1, 1, 1)
frag_atom_list = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]
ncas           = sum(ncas_f)
nelecas        = (5, 5)

CKPT_DIR = Path(__file__).parent / "ckpt" / "lassirq"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
PES_CKPT = Path(__file__).parent / "ckpt" / "pes"


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
        pr('[email] notification sent')
    except Exception as e:
        pr(f'[email] failed: {e}')


# ── Molecule + RHF ────────────────────────────────────────────────────────────
pr('=' * 70)
pr(f'CHN LASSIrq  dr={DR}  r={RS}  q={QS}')
pr(f'Checkpoint dir: {CKPT_DIR}')
pr('=' * 70)

mol = struct(DR, DR, '6-31g')
mol.output = str(CKPT_DIR / 'chn_lassirq.log')
mol.verbose  = lib.logger.INFO
mol.spin     = 8
mol.max_memory = 100000
mol.build()
mf = scf.RHF(mol)
mf.verbose = lib.logger.INFO
mf.kernel()

# ── LASSCF (from saved mo_coeff for fast convergence) ─────────────────────────
las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f)
las.verbose    = lib.logger.INFO
las.max_memory = 100000

pes_result = PES_CKPT / 'dr10_result.npz'
if pes_result.exists():
    saved_mo = np.load(str(pes_result), allow_pickle=True)['mo_coeff']
    pr('[setup] Loading saved mo_coeff from PES checkpoint')
    las.kernel(saved_mo)
else:
    pr('[setup] No PES mo_coeff — running LASSCF from localized guess')
    mo0 = las.localize_init_guess(frag_atom_list)
    las.kernel(mo0)

e_lasscf = float(las.e_states[0])
pr(f'[setup] LASSCF energy = {e_lasscf:.8f}')

# ── CASCI(10,10) reference ────────────────────────────────────────────────────
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver = csf_solver(mol, smult=1)
mc.verbose   = lib.logger.INFO
mc.kernel(las.mo_coeff)
e_casci = float(mc.e_tot)
pr(f'[setup] CASCI({ncas},{sum(nelecas)}) energy = {e_casci:.8f}')
del mc; gc.collect()

# ── Main loop ─────────────────────────────────────────────────────────────────
results = {}   # (r, q) -> (e, nroots, dt)

for r in RS:
    for q in QS:
        tag  = f'r{r}_q{q}'
        ckpt = CKPT_DIR / f'result_dr10_{tag}.npz'

        if ckpt.exists():
            d = np.load(str(ckpt), allow_pickle=True)
            e      = float(d['e'])
            nroots = int(d['nroots'])
            dt     = float(d['dt'])
            pr(f'[{tag}] loaded from ckpt: E={e:.8f}  nroots={nroots}  ({dt:.1f}s)')
            results[(r, q)] = (e, nroots, dt)
            continue

        pr(f'\n[{tag}] Running LASSIrq(r={r}, q={q}) ...')
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
            np.savez(str(ckpt),
                     e=np.array(e),
                     nroots=np.array(nroots),
                     dt=np.array(dt))
            pr(f'[{tag}] checkpoint saved')
            results[(r, q)] = (e, nroots, dt)
        except Exception as exc:
            import traceback
            dt = time() - t0
            pr(f'[{tag}] ERROR after {dt:.1f}s: {exc}')
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            results[(r, q)] = (None, None, dt)
        finally:
            try:
                del lsi, e_roots, si
            except NameError:
                pass
            gc.collect()

# ── Summary table ─────────────────────────────────────────────────────────────
pr('')
pr('=' * 70)
pr(f'LASSIrq  CHN(10,10)  dr={DR}  —  E − E_CASCI  [mH]')
pr(f'Reference:  LASSCF = {(e_lasscf - e_casci)*1000:+.4f} mH  '
   f'CASCI = {e_casci:.8f} Eh')
pr('=' * 70)
rq_label = 'r\\q'
header = f"{rq_label:>5}  " + "  ".join(f"{'q='+str(q):>10}" for q in QS)
pr(header)
pr('-' * len(header))

table_rows = []
for r in RS:
    cells = []
    for q in QS:
        res = results.get((r, q), (None, None, None))
        if res[0] is not None:
            delta = (res[0] - e_casci) * 1000
            cells.append(f'{delta:+10.4f}')
        else:
            cells.append(f"{'ERROR':>10}")
    row = f"{'r='+str(r):>5}  " + "  ".join(cells)
    pr(row)
    table_rows.append(row)

pr('=' * 70)

# ── Save results array ────────────────────────────────────────────────────────
arr = np.full((len(RS), len(QS)), np.nan)
nroots_arr = np.full((len(RS), len(QS)), -1, dtype=int)
for i, r in enumerate(RS):
    for j, q in enumerate(QS):
        res = results.get((r, q), (None, None, None))
        if res[0] is not None:
            arr[i, j] = res[0]
            nroots_arr[i, j] = res[1]

save_path = CKPT_DIR / 'lassirq_dr10.npz'
np.savez(str(save_path),
         energies=arr,
         nroots=nroots_arr,
         e_casci=np.array(e_casci),
         e_lasscf=np.array(e_lasscf),
         rs=np.array(RS),
         qs=np.array(QS),
         dr=np.array(DR))
pr(f'Results saved to {save_path}')

# ── Email notification ────────────────────────────────────────────────────────
errors_mH = (arr - e_casci) * 1000
body_lines = [
    f'LASSIrq CHN(10,10) dr={DR} complete.\n',
    f'CASCI    = {e_casci:.8f} Eh\n',
    f'LASSCF   = {e_lasscf:.8f} Eh  (ΔE = {(e_lasscf-e_casci)*1000:+.4f} mH)\n\n',
    'E − E_CASCI [mH]:\n',
    header + '\n',
    '-' * len(header) + '\n',
]
for row in table_rows:
    body_lines.append(row + '\n')

send_email('[LSI-LUSCC] LASSIrq CHN(10,10) dr=1.0 r1-5 q1-8 complete', ''.join(body_lines))
