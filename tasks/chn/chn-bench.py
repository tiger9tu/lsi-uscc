"""Benchmark lsi-luscc on the CHN (C2H4N4 / 6-31g) system.

Correct pipeline for each (m, n):
  1. Full LASSIS -> |lsi>  (done once)
  2. LSI_LUSCC(lsi, [], [], top_m=m).kernel() -> |lsi'> (sub-LASSI in top-m subspace)
  3. get_grad_exact_lassi(lsi_prime) -> excitations sorted by gradient on |lsi'>
  4. LSI_LUSCC(lsi_prime, a[:n], i[:n], top_m=m).kernel() -> lsi-luscc energy

Hyperparameters:
  m : number of dominant |las_i> components of |lsi> to keep
  n : number of top excitations by gradient magnitude on |lsi'>
"""
import numpy as np
from pyscf import gto, scf, lib, mcscf, fci
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.exploratory.citools.grad import get_grad_exact_lassi
from mrh.my_pyscf import lassi
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from lcc import LSI_LUSCC
from helper.util import get_sorted_excitations
from time import time

# ── 1. Hyperparameter sweep grid ───────────────────────────────────────────
M_VALUES = [1, 2, 4, 8, 16, 32]   # number of dominant LAS components to keep
N_VALUES = [5, 10, 25, 50, 100, 200, 500]  # number of excitations to include

# ── 2. System parameters ───────────────────────────────────────────────────
ncas_f         = (4, 2, 4)
nelecas_f      = ((2, 2), (1, 1), (2, 2))
spin_sub_f     = (1, 1, 1)
frag_atom_list = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]

# ── 3. Molecule + RHF ──────────────────────────────────────────────────────
lib.logger.TIMER_LEVEL = lib.logger.INFO
mol = struct(1.0, 1.0, '6-31g')
mol.output = 'chn_bench.log'
mol.verbose = 3
mol.spin = 8
mol.max_memory = 16000
mol.build()
mf = scf.RHF(mol).run()

ncas = sum(ncas_f)
nelecas = (5, 5)

# ── 4. Reference: CASCI ────────────────────────────────────────────────────
mc = mcscf.CASCI(mf, ncas, nelecas).set(fcisolver=csf_solver(mol, smult=1))
mc.kernel()
e_casci = mc.e_tot
print(f"CASCI(10,10) energy = {e_casci:.8f}", flush=True)

cas_rdm3s = fci.direct_spin1.make_rdm123s(mc.ci, ncas, nelecas)

# ── 5. LASSCF ──────────────────────────────────────────────────────────────
las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f)
mo_coeff = las.localize_init_guess(frag_atom_list)
las.kernel(mo_coeff)
e_las = las.e_tot
print(f"LASSCF energy       = {e_las:.8f}", flush=True)

# ── 6. Full LASSIS ─────────────────────────────────────────────────────────
lsi = lassi.LASSIS(las)
t0 = time()
e_lsi, si = lsi.kernel()
t_lsi = time() - t0
e_lassis = e_lsi[0]
print(f"LASSIS energy       = {e_lassis:.8f}  ({t_lsi:.1f} s)", flush=True)
print(f"LASSIS nroots       = {len(e_lsi)}", flush=True)

si_vec = lsi.si[:, 0]
order = np.argsort(-np.abs(si_vec))
print("Top-10 |ci| coefficients:", flush=True)
for k in order[:10]:
    print(f"  j={k:4d}  ci={si_vec[k]:+.4f}", flush=True)

# ── 7. Sweep ───────────────────────────────────────────────────────────────
print("\n" + "="*80, flush=True)
print(f"{'m':>4}  {'n':>5}  {'E(lsi-luscc)':>14}  {'ΔE_casci(mH)':>13}  "
      f"{'t_lsi_prime(s)':>14}  {'t_luscc(s)':>10}", flush=True)
print("="*80, flush=True)

for m in M_VALUES:
    # ── Step 2: sub-LASSI in top-m subspace -> lsi_prime ──────────────────
    t0 = time()
    lsi_prime = LSI_LUSCC(lsi, [], [], top_m=m)
    e_prime, _ = lsi_prime.kernel()
    t_prime = time() - t0
    e_lsi_prime = e_prime[0]
    print(f"\n[m={m}] |lsi'> energy = {e_lsi_prime:.8f}  ({t_prime:.1f} s)", flush=True)

    # ── Step 3: gradients on |lsi'> ───────────────────────────────────────
    _, g_sel, a_idxs_all, i_idxs_all = get_grad_exact_lassi(lsi_prime, state=0)
    g_all = np.array([g for g, _ in g_sel])
    a_sorted, i_sorted, _ = get_sorted_excitations(
        a_idxs_all, i_idxs_all, g_all, fraction=1.0)
    n_total = len(a_sorted)
    print(f"[m={m}] excitations available: {n_total}", flush=True)

    # ── Step 4: LSI-LUSCC for each n ──────────────────────────────────────
    for n in N_VALUES:
        n_use = min(n, n_total)
        a_use = a_sorted[:n_use]
        i_use = i_sorted[:n_use]

        t0 = time()
        try:
            luscc = LSI_LUSCC(lsi_prime, a_use, i_use, top_m=m)
            e_roots, _ = luscc.kernel()
            e = e_roots[0]
            dt = time() - t0
            delta_mh = (e - e_casci) * 1000
            print(f"{m:>4}  {n_use:>5}  {e:>14.8f}  {delta_mh:>+13.4f}  "
                  f"{t_prime:>14.1f}  {dt:>10.1f}", flush=True)
        except Exception as exc:
            dt = time() - t0
            print(f"{m:>4}  {n_use:>5}  {'ERROR':>14}  {str(exc)[:40]}  {dt:.1f}s", flush=True)

print("\n" + "="*80, flush=True)
print(f"Reference energies:", flush=True)
print(f"  CASCI  = {e_casci:.8f}", flush=True)
print(f"  LASSIS = {e_lassis:.8f}  (ΔE = {(e_lassis-e_casci)*1000:+.4f} mH)", flush=True)
print(f"\nChemical accuracy threshold: ±1.594 mH", flush=True)
