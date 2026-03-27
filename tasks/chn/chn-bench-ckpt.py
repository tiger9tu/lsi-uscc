"""Benchmark lsi-luscc on the CHN (C2H4N4 / 6-31g) system — with checkpointing.

Correct pipeline for each (m, n):
  1. Full LASSIS -> |lsi>  (done once)
  2. LSI_LUSCC(lsi, [], [], top_m=m).kernel() -> |lsi'> (sub-LASSI in top-m subspace)
  3. get_grad_exact_lassi(lsi_prime) -> excitations sorted by gradient on |lsi'>
  4. LSI_LUSCC(lsi_prime, a[:n], i[:n], top_m=m).kernel() -> lsi-luscc energy

Checkpointing strategy
----------------------
PySCF objects (mf, las, lsi, lsi_prime) cannot be reliably pickled because they
contain locally-defined FCI solvers.  The SCF/LASSCF/LASSIS steps are fast
(seconds to a few minutes) so they are re-run on every invocation.

Only the expensive outputs are checkpointed:
  ckpt/grad_m{m}.npz     -- gradient arrays for each m (~10 min each)
  ckpt/result_m{m}_n{n}.npz -- per-(m,n) LUSCC ground-state energy

Delete ckpt/ (or individual files) to force recomputation.
"""
import numpy as np
from pathlib import Path
from time import time

# ── Checkpoint helpers ─────────────────────────────────────────────────────────
CKPT_DIR = Path(__file__).parent / "ckpt"
CKPT_DIR.mkdir(exist_ok=True)

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

# ── Imports ────────────────────────────────────────────────────────────────────
from pyscf import gto, scf, lib, mcscf, fci
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.exploratory.citools.grad import get_grad_exact_lassi
from mrh.my_pyscf import lassi
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from lcc import LSI_LUSCC
from helper.util import get_sorted_excitations

# ── 1. Hyperparameter sweep grid ───────────────────────────────────────────────
# N_VALUES is capped per m: small m converges slowly so we stop early;
# large m can afford more excitations since each step is cheaper relative to gain.
CHEM_ACC_MH   = 1.594   # ±1.594 mH = chemical accuracy threshold

M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]

N_VALUES_BY_M = {
    1:   [5, 10, 25, 50, 100, 200, 500],
    2:   [5, 10, 25, 50, 100, 200, 500],
    4:   [5, 10, 25, 50, 100, 200, 500],
    8:   [5, 10, 25, 50, 100, 200, 500, 1000],
    16:  [5, 10, 25, 50, 100, 200, 500, 1000],
    32:  [5, 10, 25, 50, 100, 200, 500, 1000],
    64:  [5, 10, 25, 50, 100, 200, 500, 1000],
    128: [5, 10, 25, 50, 100, 200, 500, 1000],
}

# ── 2. System parameters ────────────────────────────────────────────────────────
ncas_f         = (4, 2, 4)
nelecas_f      = ((2, 2), (1, 1), (2, 2))
spin_sub_f     = (1, 1, 1)
frag_atom_list = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]

# ── 3. Molecule + RHF ──────────────────────────────────────────────────────────
lib.logger.TIMER_LEVEL = lib.logger.INFO
mol = struct(1.0, 1.0, '6-31g')
mol.output = str(CKPT_DIR / 'chn_bench.log')
mol.verbose = 3
mol.spin = 8
mol.max_memory = 16000
mol.build()
mf = scf.RHF(mol).run()

ncas    = sum(ncas_f)
nelecas = (5, 5)

# ── 4. CASCI reference ─────────────────────────────────────────────────────────
mc = mcscf.CASCI(mf, ncas, nelecas).set(fcisolver=csf_solver(mol, smult=1))
mc.kernel()
e_casci = mc.e_tot
print(f"CASCI(10,10) energy = {e_casci:.8f}", flush=True)

# ── 5. LASSCF ──────────────────────────────────────────────────────────────────
las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f)
mo_coeff = las.localize_init_guess(frag_atom_list)
las.kernel(mo_coeff)
print(f"LASSCF energy       = {las.e_tot:.8f}", flush=True)

# ── 6. Full LASSIS ─────────────────────────────────────────────────────────────
lsi = lassi.LASSIS(las)
t0 = time()
e_lsi, si = lsi.kernel()
t_lsi = time() - t0
e_lassis = float(e_lsi[0])
print(f"LASSIS energy       = {e_lassis:.8f}  ({t_lsi:.1f} s)", flush=True)
print(f"LASSIS nroots       = {len(e_lsi)}", flush=True)

si_vec = lsi.si[:, 0]
order  = np.argsort(-np.abs(si_vec))
print("Top-10 |ci| coefficients:", flush=True)
for k in order[:10]:
    print(f"  j={k:4d}  ci={si_vec[k]:+.4f}", flush=True)

# ── 7. Sweep ────────────────────────────────────────────────────────────────────
print("\n" + "="*80, flush=True)
print(f"{'m':>4}  {'n':>5}  {'E(lsi-luscc)':>14}  {'ΔE_casci(mH)':>13}  "
      f"{'t_lsi_prime(s)':>14}  {'t_luscc(s)':>10}", flush=True)
print("="*80, flush=True)

for m in M_VALUES:
    # ── Step 2: sub-LASSI in top-m subspace -> lsi_prime ──────────────────────
    t0 = time()
    lsi_prime = LSI_LUSCC(lsi, [], [], top_m=m)
    e_prime, _ = lsi_prime.kernel()
    t_prime = time() - t0
    e_lsi_prime = float(e_prime[0])
    print(f"\n[m={m}] |lsi'> energy = {e_lsi_prime:.8f}  ({t_prime:.1f} s)", flush=True)

    # ── Step 3: gradients on |lsi'> (checkpoint this expensive step) ──────────
    grad_ckpt = load_npz(f"grad_m{m}.npz")

    if grad_ckpt is None:
        print(f"[m={m}] computing gradients (may take ~10 min) ...", flush=True)
        t_g0 = time()
        _, g_sel, a_idxs_all, i_idxs_all = get_grad_exact_lassi(lsi_prime, state=0)
        t_grad = time() - t_g0
        g_all = np.array([g for g, _ in g_sel])
        a_sorted, i_sorted, _ = get_sorted_excitations(
            a_idxs_all, i_idxs_all, g_all, fraction=1.0)
        n_total = len(a_sorted)
        save_npz(f"grad_m{m}.npz",
                 a_sorted=np.array(a_sorted, dtype=object),
                 i_sorted=np.array(i_sorted, dtype=object),
                 g_all=g_all,
                 n_total=np.array(n_total),
                 t_grad=np.array(t_grad))
        print(f"[m={m}] gradients done in {t_grad:.1f} s, "
              f"excitations available: {n_total}", flush=True)
    else:
        a_sorted = list(grad_ckpt["a_sorted"])
        i_sorted = list(grad_ckpt["i_sorted"])
        g_all    = grad_ckpt["g_all"]
        n_total  = int(grad_ckpt["n_total"])
        t_grad   = float(grad_ckpt["t_grad"])
        print(f"[m={m}] excitations available: {n_total}  "
              f"(grad took {t_grad:.1f} s) [from ckpt]", flush=True)

    # ── Step 4: LSI-LUSCC for each n ──────────────────────────────────────────
    reached_chem_acc = False
    for n in N_VALUES_BY_M.get(m, N_VALUES_BY_M[128]):
        if reached_chem_acc:
            break
        res_ckpt = load_npz(f"result_m{m}_n{n}.npz")
        n_use    = min(n, n_total)

        if res_ckpt is not None:
            e  = float(res_ckpt["e"])
            dt = float(res_ckpt["dt"])
            delta_mh = (e - e_casci) * 1000
            print(f"{m:>4}  {n_use:>5}  {e:>14.8f}  {delta_mh:>+13.4f}  "
                  f"{t_prime:>14.1f}  {dt:>10.1f}  [from ckpt]", flush=True)
            if abs(delta_mh) <= CHEM_ACC_MH:
                print(f"  *** Chemical accuracy reached at m={m}, n={n_use}! ***", flush=True)
                reached_chem_acc = True
            continue

        a_use = a_sorted[:n_use]
        i_use = i_sorted[:n_use]

        t0 = time()
        try:
            luscc    = LSI_LUSCC(lsi_prime, a_use, i_use, top_m=m)
            e_roots, _ = luscc.kernel()
            e  = float(e_roots[0])
            dt = time() - t0
            delta_mh = (e - e_casci) * 1000
            print(f"{m:>4}  {n_use:>5}  {e:>14.8f}  {delta_mh:>+13.4f}  "
                  f"{t_prime:>14.1f}  {dt:>10.1f}", flush=True)
            save_npz(f"result_m{m}_n{n}.npz", e=np.array(e), dt=np.array(dt))
            if abs(delta_mh) <= CHEM_ACC_MH:
                print(f"  *** Chemical accuracy reached at m={m}, n={n_use}! ***", flush=True)
                reached_chem_acc = True
        except Exception as exc:
            dt = time() - t0
            print(f"{m:>4}  {n_use:>5}  {'ERROR':>14}  {str(exc)[:40]}  {dt:.1f}s",
                  flush=True)

print("\n" + "="*80, flush=True)
print("Reference energies:", flush=True)
print(f"  CASCI  = {e_casci:.8f}", flush=True)
print(f"  LASSIS = {e_lassis:.8f}  (ΔE = {(e_lassis-e_casci)*1000:+.4f} mH)", flush=True)
print(f"\nChemical accuracy threshold: ±1.594 mH", flush=True)
