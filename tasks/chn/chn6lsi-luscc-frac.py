"""LSI-LUSCC on C2H4N4 / 6-31g with 6-orbital active space, m=2, fraction sweep.

Setup mirrors chn6lassis.py:
  - LASSCF((3,3), ((2,1),(1,2))), 2 state-averaged roots
  - sort_mo([16,18,22,23,24,26])
  - localize_init_guess on fragment atom ranges

Pipeline for each frac in [0.1, 0.2, ..., 1.0]:
  1. Full LASSIS -> |lsi>  (done once)
  2. LSI_LUSCC(lsi, [], [], top_m=2).kernel() -> |lsi'> (sub-LASSI in top-2 subspace)
  3. get_grad_exact_lassi(lsi_prime) -> all excitations sorted by gradient (checkpointed)
  4. LSI_LUSCC(lsi_prime, a[:n], i[:n], top_m=2).kernel() for n = frac * n_total
"""
import gc
import numpy as np
from pathlib import Path
from time import time

# ── Checkpoint helpers ──────────────────────────────────────────────────────────
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

# ── Imports ─────────────────────────────────────────────────────────────────────
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.citools.grad import get_grad_exact_lassi
from mrh.my_pyscf import lassi
from mrh.my_pyscf.tools import molden
from c2h4n4_struct import structure as struct
from lcc import LSI_LUSCC
from helper.util import get_sorted_excitations

# ── Parameters ──────────────────────────────────────────────────────────────────
M = 2
FRACS = np.round(np.arange(0.1, 1.01, 0.1), 2)

# ── Molecule + RHF ─────────────────────────────────────────────────────────────
mol = struct(0, 0, '6-31g')
mol.output = 'c2h4n4_lsi_luscc_m2_631g.log'
mol.verbose = lib.logger.INFO
mol.build()
mf = scf.RHF(mol).run()

# ── LASSCF ─────────────────────────────────────────────────────────────────────
las = LASSCF(mf, (3, 3), ((2, 1), (1, 2)))
las = las.state_average(
    [0.5, 0.5],
    spins=[[1, -1], [-1, 1]],
    smults=[[2, 2], [2, 2]],
    charges=[[0, 0], [0, 0]])
mo = las.sort_mo([16, 18, 22, 23, 24, 26])
mo = las.localize_init_guess((list(range(5)), list(range(5, 10))), mo)
las.kernel(mo)
print(f"LASSCF((3,3),(3,3)) energy = {las.e_tot}", flush=True)

# ── CASCI reference ─────────────────────────────────────────────────────────────
mc = mcscf.CASCI(mf, 6, 6).set(fcisolver=csf_solver(mol, smult=1))
mc.kernel(las.mo_coeff)
e_casci = mc.e_tot
print(f"CASCI(6,6) energy = {e_casci:.8f}", flush=True)
del mc
gc.collect()

# ── Full LASSIS ─────────────────────────────────────────────────────────────────
lsi = lassi.LASSIS(las)
t0 = time()
e_lsi, si = lsi.kernel()
t_lsi = time() - t0
e_lassis = float(e_lsi[0])
print(f"LASSIS energy = {e_lassis:.8f}  ({t_lsi:.1f} s)", flush=True)
print(f"LASSIS nroots = {len(e_lsi)}", flush=True)
molden.from_lassi(las, 'c2h4n4_lassis_631g.molden', si=si)

# ── Step 2: sub-LASSI in top-m=2 subspace -> lsi_prime ────────────────────────
t0 = time()
lsi_prime = LSI_LUSCC(lsi, [], [], top_m=M)
e_prime, _si = lsi_prime.kernel()
t_prime = time() - t0
e_lsi_prime = float(e_prime[0])
del e_prime, _si
print(f"\n[m={M}] |lsi'> energy = {e_lsi_prime:.8f}  ({t_prime:.1f} s)", flush=True)

# ── Step 3: gradients on |lsi'> (checkpointed) ────────────────────────────────
grad_ckpt = load_npz(f"grad_m{M}_chn6.npz")

if grad_ckpt is None:
    print(f"[m={M}] computing gradients ...", flush=True)
    t_g0 = time()
    _, g_sel, a_idxs_all, i_idxs_all = get_grad_exact_lassi(lsi_prime, state=0)
    t_grad = time() - t_g0
    g_all = np.array([g for g, _ in g_sel])
    a_sorted, i_sorted, _ = get_sorted_excitations(
        a_idxs_all, i_idxs_all, g_all, fraction=1.0)
    n_total = len(a_sorted)
    save_npz(f"grad_m{M}_chn6.npz",
             a_sorted=np.array(a_sorted, dtype=object),
             i_sorted=np.array(i_sorted, dtype=object),
             g_all=g_all,
             n_total=np.array(n_total),
             t_grad=np.array(t_grad))
    print(f"[m={M}] gradients done in {t_grad:.1f} s, "
          f"excitations available: {n_total}", flush=True)
else:
    a_sorted = list(grad_ckpt["a_sorted"])
    i_sorted = list(grad_ckpt["i_sorted"])
    g_all    = grad_ckpt["g_all"]
    n_total  = int(grad_ckpt["n_total"])
    t_grad   = float(grad_ckpt["t_grad"])
    print(f"[m={M}] excitations available: {n_total}  "
          f"(grad took {t_grad:.1f} s) [from ckpt]", flush=True)
del grad_ckpt
gc.collect()

# ── Step 4: fraction sweep ─────────────────────────────────────────────────────
print("\n" + "="*80, flush=True)
print(f"{'frac':>6}  {'n_exc':>6}  {'E(lsi-luscc)':>14}  "
      f"{'ΔE_casci(mH)':>13}  {'t(s)':>8}", flush=True)
print("="*80, flush=True)

results = []
for frac in FRACS:
    n_use = max(1, int(round(frac * n_total)))
    res_ckpt = load_npz(f"result_m{M}_frac{frac:.2f}_chn6.npz")

    if res_ckpt is not None:
        e  = float(res_ckpt["e"])
        dt = float(res_ckpt["dt"])
        delta_mh = (e - e_casci) * 1000
        print(f"{frac:>6.2f}  {n_use:>6d}  {e:>14.8f}  "
              f"{delta_mh:>+13.4f}  {dt:>8.2f}  [from ckpt]", flush=True)
        results.append((frac, n_use, e, dt))
        continue

    a_use = a_sorted[:n_use]
    i_use = i_sorted[:n_use]

    t0 = time()
    try:
        luscc = LSI_LUSCC(lsi_prime, a_use, i_use, top_m=M)
        e_roots, _ = luscc.kernel()
        e  = float(e_roots[0])
        dt = time() - t0
        delta_mh = (e - e_casci) * 1000
        print(f"{frac:>6.2f}  {n_use:>6d}  {e:>14.8f}  "
              f"{delta_mh:>+13.4f}  {dt:>8.2f}", flush=True)
        save_npz(f"result_m{M}_frac{frac:.2f}_chn6.npz",
                 e=np.array(e), dt=np.array(dt))
        results.append((frac, n_use, e, dt))
    except Exception as exc:
        dt = time() - t0
        print(f"{frac:>6.2f}  {n_use:>6d}  {'ERROR':>14}  "
              f"{str(exc)[:40]}  {dt:.2f}s", flush=True)
    finally:
        try:
            del luscc, e_roots
        except NameError:
            pass
        gc.collect()

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*80, flush=True)
print("Reference energies:", flush=True)
print(f"  CASCI(6,6)  = {e_casci:.8f}", flush=True)
print(f"  LASSIS      = {e_lassis:.8f}  (ΔE = {(e_lassis-e_casci)*1000:+.4f} mH)", flush=True)
print(f"  |lsi'>(m=2) = {e_lsi_prime:.8f}  (ΔE = {(e_lsi_prime-e_casci)*1000:+.4f} mH)", flush=True)
print(f"\nChemical accuracy threshold: ±1.594 mH", flush=True)

if results:
    arr = np.array([(r[0], r[1], r[2], r[3]) for r in results])
    np.save('chn6_lsi_luscc_m2_frac_sweep.npy', arr)
    print("\nResults saved to chn6_lsi_luscc_m2_frac_sweep.npy", flush=True)
    print("Columns: fraction, n_excitations, e_gs, time_s", flush=True)
