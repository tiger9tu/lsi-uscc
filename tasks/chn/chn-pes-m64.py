"""Compute CHN (C2H4N4 / 6-31g) potential energy surface for
LASSCF / CASCI / LASSIS / LSI-LUSCC.

Scans the N-N bond displacement dr from 2.0 → 0.0 Å in steps of 0.1 Å (21 pts).
Fixed hyperparameters: m = 64 (dominant LAS components), n = 66 (excitations).

Pipeline at each geometry
-------------------------
1. LASSCF + LASSIS  — via as_scanner() (carries MO from previous point)
2. CASCI            — reference energy at LASSCF MOs
3. LSI-LUSCC step 1 — LSI_LUSCC(lsi, [], [], top_m=M) -> lsi_prime
4. Gradients        — get_grad_exact_lassi(lsi_prime)         [~10 min]
5. LSI-LUSCC step 2 — LSI_LUSCC(lsi_prime, a[:N], i[:N], top_m=M)  [hours]

Checkpointing
-------------
ckpt/pes_m64/dr{ix:02d}_grad.npz    saved after gradient step (before lsi-luscc)
ckpt/pes_m64/dr{ix:02d}_result.npz  saved after lsi-luscc (includes all 5 energies)

Restart behaviour
-----------------
The LASSIS scanner is always re-run from dr=2.0 forward (LASSCF+LASSIS per point
takes minutes; lsi-luscc takes hours, so this overhead is acceptable).
For points whose result.npz already exists the lsi-luscc step is skipped.
For points whose grad.npz already exists the gradient step is skipped.
"""

import gc
import numpy as np
from pathlib import Path
from time import time

# ── Checkpoint helpers ─────────────────────────────────────────────────────
CKPT_DIR = Path(__file__).parent / "ckpt" / "pes_m64"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def save_npz(name, **arrays):
    path = CKPT_DIR / name
    np.savez(str(path), **arrays)
    print(f"[ckpt] saved {path}", flush=True)

def load_npz(name):
    path = CKPT_DIR / name
    if path.exists():
        print(f"[ckpt] loading {path}", flush=True)
        return np.load(str(path), allow_pickle=True)
    return None

# ── Imports ────────────────────────────────────────────────────────────────
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.exploratory.citools.grad import get_grad_exact_lassi
from mrh.my_pyscf.lassi.lassis import LASSIS
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from lcc import LSI_LUSCC
from helper.util import get_sorted_excitations

# ── Parameters ────────────────────────────────────────────────────────────
M = 64      # number of dominant LAS components
N = 66      # number of excitations

ncas_f         = (4, 2, 4)
nelecas_f      = ((2, 2), (1, 1), (2, 2))
spin_sub_f     = (1, 1, 1)
frag_atom_list = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]
ncas           = sum(ncas_f)
nelecas        = (5, 5)

# ── Scan grid ─────────────────────────────────────────────────────────────
# dr=2.0 (reference) then 1.9, 1.8, … 0.0  →  21 points total
scan_drs   = [2.0] + [round(2.0 + (i+1)*(-0.1), 1) for i in range(20)]
# ix = int(round(dr * 10))  →  20, 19, 18, … 0

lib.logger.TIMER_LEVEL = lib.logger.INFO

# ── Reference geometry (dr=2.0) ───────────────────────────────────────────
print("=" * 70, flush=True)
print(f"CHN PES  m={M}  n={N}  scan dr: {scan_drs[0]} → {scan_drs[-1]}", flush=True)
print(f"Checkpoint dir: {CKPT_DIR}", flush=True)
print("=" * 70, flush=True)

mol = struct(2.0, 2.0, '6-31g')
mol.output = str(CKPT_DIR / 'chn_pes.log')
mol.verbose = lib.logger.INFO
mol.spin = 8
mol.max_memory = 100000
mol.build()
mf = scf.RHF(mol).run()

las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f)
mo_coeff = las.localize_init_guess(frag_atom_list)
las.kernel(mo_coeff)
print(f"[dr=2.0] LASSCF energy = {las.e_states[0]:.8f}", flush=True)

lsi = LASSIS(las)
lsi.kernel()
print(f"[dr=2.0] LASSIS energy = {lsi.e_roots[0]:.8f}", flush=True)

# ── Convert to scanner (carries MO from point to point) ───────────────────
lsi = lsi.as_scanner()

# ── Helper: run lsi-luscc for one point ───────────────────────────────────
def run_luscc_point(dr, lsi_obj):
    """Run CASCI + lsi-luscc at the current scanner state. Save checkpoint."""
    ix = int(round(dr * 10))
    e_lasscf  = lsi_obj._las.e_states[0]
    e_lassis  = lsi_obj.e_roots[0]
    mol_cur   = lsi_obj.mol

    # CASCI reference
    mc = mcscf.CASCI(lsi_obj._las._scf, ncas, nelecas).set(
        fcisolver=csf_solver(mol_cur, smult=1))
    mc.kernel(lsi_obj.mo_coeff)
    e_casci = mc.e_tot
    del mc
    gc.collect()
    print(f"[dr={dr:.1f}] CASCI  = {e_casci:.8f}", flush=True)

    # LSI-LUSCC step 1: lsi_prime
    lsi_prime = LSI_LUSCC(lsi_obj, [], [], top_m=M)
    e_prime_arr, _ = lsi_prime.kernel()
    e_lsi_prime = float(e_prime_arr[0])
    print(f"[dr={dr:.1f}] lsi'   = {e_lsi_prime:.8f}", flush=True)

    # Gradients (checkpoint if available)
    grad_ckpt = load_npz(f"dr{ix:02d}_grad.npz")
    if grad_ckpt is None:
        t_g0 = time()
        print(f"[dr={dr:.1f}] computing gradients...", flush=True)
        _, g_sel, a_all, i_all = get_grad_exact_lassi(lsi_prime, state=0)
        t_grad = time() - t_g0
        g_all_arr = np.array([g for g, _ in g_sel])
        a_sorted, i_sorted, _ = get_sorted_excitations(
            a_all, i_all, g_all_arr, fraction=1.0)
        n_total = len(a_sorted)
        save_npz(f"dr{ix:02d}_grad.npz",
                 a_sorted=np.array(a_sorted, dtype=object),
                 i_sorted=np.array(i_sorted, dtype=object),
                 g_all=g_all_arr,
                 n_total=np.array(n_total),
                 t_grad=np.array(t_grad))
        del g_sel, a_all, i_all, g_all_arr
        gc.collect()
        print(f"[dr={dr:.1f}] gradients done in {t_grad:.1f}s, "
              f"n_total={n_total}", flush=True)
    else:
        a_sorted = list(grad_ckpt["a_sorted"])
        i_sorted = list(grad_ckpt["i_sorted"])
        n_total  = int(grad_ckpt["n_total"])
        print(f"[dr={dr:.1f}] gradients loaded from ckpt, "
              f"n_total={n_total}", flush=True)
    del grad_ckpt
    gc.collect()

    # LSI-LUSCC step 2
    n_use = min(N, n_total)
    print(f"[dr={dr:.1f}] running LSI-LUSCC m={M} n={n_use}...", flush=True)
    t0 = time()
    e_luscc = float('nan')
    try:
        luscc = LSI_LUSCC(lsi_prime, a_sorted[:n_use], i_sorted[:n_use], top_m=M)
        e_roots, _ = luscc.kernel()
        e_luscc = float(e_roots[0])
        dt = time() - t0
        print(f"[dr={dr:.1f}] LSI-LUSCC = {e_luscc:.8f}  "
              f"ΔE(CASCI)={( e_luscc - e_casci)*1000:+.4f} mH  "
              f"ΔE(LASSIS)={(e_luscc - e_lassis)*1000:+.4f} mH  "
              f"({dt:.1f} s)", flush=True)
    except Exception as exc:
        dt = time() - t0
        print(f"[dr={dr:.1f}] LSI-LUSCC ERROR: {exc}  ({dt:.1f} s)", flush=True)
    finally:
        try:
            del luscc, e_roots
        except NameError:
            pass
        gc.collect()

    # Save result
    save_npz(f"dr{ix:02d}_result.npz",
             dr=np.array(dr),
             e_lasscf=np.array(e_lasscf),
             e_casci=np.array(e_casci),
             e_lassis=np.array(e_lassis),
             e_lsi_prime=np.array(e_lsi_prime),
             e_luscc=np.array(e_luscc),
             mo_coeff=lsi_obj.mo_coeff)

    del lsi_prime, a_sorted, i_sorted
    gc.collect()


# ── Process dr=2.0 (already computed above, no scanner call needed) ────────
ix_ref = int(round(2.0 * 10))
if load_npz(f"dr{ix_ref:02d}_result.npz") is None:
    run_luscc_point(2.0, lsi)
else:
    print(f"[dr=2.0] result already in ckpt, skipping lsi-luscc", flush=True)

# ── Scan dr = 1.9 → 0.0 ──────────────────────────────────────────────────
fmt_row = "{:>5.1f}  {:>14.8f}  {:>14.8f}  {:>14.8f}  {:>14.8f}"

for dr in scan_drs[1:]:
    ix = int(round(dr * 10))
    print(f"\n{'='*70}", flush=True)
    print(f"[dr={dr:.1f}] ix={ix}", flush=True)
    t_pt = time()

    mol_i = struct(dr, dr, '6-31g')
    mol_i.max_memory = mol.max_memory
    lsi(mol_i)   # scanner: runs LASSCF + LASSIS, updates lsi.mo_coeff etc.

    print(f"[dr={dr:.1f}] LASSCF = {lsi._las.e_states[0]:.8f}", flush=True)
    print(f"[dr={dr:.1f}] LASSIS = {lsi.e_roots[0]:.8f}", flush=True)

    if load_npz(f"dr{ix:02d}_result.npz") is not None:
        print(f"[dr={dr:.1f}] result in ckpt, skipping lsi-luscc", flush=True)
    else:
        run_luscc_point(dr, lsi)

    print(f"[dr={dr:.1f}] point done ({time()-t_pt:.1f} s total)", flush=True)

# ── Summary table ─────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
print(f"Summary  (m={M}, n={N})", flush=True)
print(f"{'dr':>5}  {'E(LASSCF)':>14}  {'E(CASCI)':>14}  "
      f"{'E(LASSIS)':>14}  {'E(lsi-luscc)':>14}", flush=True)
print("-" * 70, flush=True)
for dr in scan_drs:
    ix = int(round(dr * 10))
    ckpt = load_npz(f"dr{ix:02d}_result.npz")
    if ckpt is not None:
        print(fmt_row.format(
            dr,
            float(ckpt["e_lasscf"]),
            float(ckpt["e_casci"]),
            float(ckpt["e_lassis"]),
            float(ckpt["e_luscc"])), flush=True)
    else:
        print(f"{dr:>5.1f}  {'(not done)':>14}", flush=True)
