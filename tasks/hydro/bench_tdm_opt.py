"""Benchmark: Optimization 2 (within-group spectator TDM caching).

The optimization only fires when the reference LASSI has lroots > 1 in some
rootspace, so that multiple sig_indices j1, j2 map to the same rootspace r and
form within-group excited-state pairs.  We create that scenario using
all_single_excitations + lasci(lroots) on H6, then set up a multi-root LASSI
reference and compare LSI-LUSCC timing with/without the optimization.

Run from the lsi-uscc root:
    cd /home/tuyue/workspace/lsi-uscc
    python tasks/hydro/bench_tdm_opt.py
"""
import sys, time, types
import numpy as np
from pathlib import Path
from pyscf import gto, scf, lib
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSI
from mrh.my_pyscf.lassi.spaces import all_single_excitations
from mrh.my_pyscf.mcscf.lasci import get_space_info
from mrh.exploratory.citools.grad import get_grad_exact_lassi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lcc.solver import LSI_LUSCC, LSIFragTDMInt

# ── 1. Molecule + RHF ─────────────────────────────────────────────────────────
pwd = Path(__file__).resolve().parent
geom_path = pwd.parent / 'geom' / 'h6.xyz'
with geom_path.open() as f:
    xyz = f.read()

mol = gto.M(atom=xyz, basis='sto-3g', output='/dev/null', verbose=0)
mf = scf.RHF(mol).run()
print(f"RHF energy = {mf.e_tot:.8f}")

# ── 2. LASSCF (3 frags, 2 orbs each) ─────────────────────────────────────────
las0 = LASSCF(mf, (2, 2, 2), (2, 2, 2), spin_sub=(1, 1, 1), verbose=0)
mo = las0.localize_init_guess([(0, 1), (2, 3), (4, 5)], mf.mo_coeff)
las0.kernel(mo)
print(f"LASSCF energy = {las0.e_tot:.8f}")

# ── 3. Expand space: all single excitations → many rootspaces ─────────────────
las1 = las0
for _ in range(2):
    las1 = all_single_excitations(las1)
print(f"Number of rootspaces after excitations: {las1.nroots}")

# Set lroots > 1 only for neutral-singlet rootspaces (safe for any 2e/2orb fragment).
# Charged or non-singlet rootspaces are left at lroots=1.
charges, spins, smults, _ = get_space_info(las1)
# 4 - smult: singlet→3, doublet→2, triplet→1 (same formula as in mrh tests)
lroots = 4 - smults
# Charged fragments within a rootspace can have very few CSFs; cap them to 1.
lroots[(charges != 0) | (lroots < 1)] = 1
print(f"lroots per fragment: min={lroots.min()}, max={lroots.max()}, "
      f"rootspaces with lroots>1: {np.any(lroots > 1, axis=1).sum()}/{len(lroots)}")

las1.conv_tol_grad = las1.conv_tol_self = 9e99   # skip orbital optimization
las1.lasci(lroots=lroots.T)
print(f"LASCI done.  Total CI roots: {np.sum(lroots)}")

# ── 4. Multi-root LASSI reference ─────────────────────────────────────────────
lsi = LASSI(las1, verbose=0)
lsi.kernel()
print(f"LASSI done.  nstates = {lsi.si.shape[0]}")

# ── 5. Select excitations from the LASSI ground state ────────────────────────
_, g_sel, a_idxs, i_idxs = get_grad_exact_lassi(lsi, state=0, epsilon=0.0)
order = np.argsort(-np.abs(np.array(g_sel)[:, 0]))
a_idxs = [a_idxs[k] for k in order]
i_idxs = [i_idxs[k] for k in order]
print(f"Total operators available: {len(a_idxs)}")
# Limit to top-K for a tractable but instructive benchmark
K = 40
a_idxs, i_idxs = a_idxs[:K], i_idxs[:K]
print(f"Using top-{K} operators for benchmark")

# ── 6. Top-m selection ────────────────────────────────────────────────────────
TOP_M = 6   # selects 6 product states from the reference LASSI

# We verify how many selected states share a rootspace (determines group size)
from mrh.my_pyscf.lassi.citools import get_lroots, get_rootaddr_fragaddr
lroots_ref = get_lroots(lsi.ci)
rootaddr, fragaddr = get_rootaddr_fragaddr(lroots_ref)
si_vec = lsi.si[:, 0]
top_idxs = np.argsort(-np.abs(si_vec))[:TOP_M]
r_per_idx = [rootaddr[j] for j in top_idxs]
from collections import Counter
r_counts = Counter(r_per_idx)
print(f"\nTop-{TOP_M} states rootspace distribution: {dict(r_counts)}")
max_group_size = max(r_counts.values())
n_within_pairs = sum(c*(c-1)//2 for c in r_counts.values())
print(f"Within-group pairs per operator direction: {n_within_pairs}")
print(f"Spectator fragments per pair: {las1.nfrags - 2}  (operators act on 2 frags)")
print(f"Expected trans_rdm12s calls saved: ~{n_within_pairs * (las1.nfrags-2) * len(a_idxs)} "
      f"(for spectator pairs, before shared-block amortization)")


# ── 7. Helper: NoOpt subclass ─────────────────────────────────────────────────
class LSI_LUSCC_NoOpt(LSI_LUSCC):
    """Identical to LSI_LUSCC but skips the TDM cache optimization."""
    def _build_lsi_tdm_cache(self):
        return None


# ── 8. Timing comparison ──────────────────────────────────────────────────────
N_REPEAT = 3

def run_and_time(cls, label):
    times = []
    energy = None
    for rep in range(N_REPEAT):
        obj = cls(lsi, a_idxs, i_idxs, top_m=TOP_M, verbose=0)
        obj.max_memory = 20000   # allow up to 20 GB
        t0 = time.perf_counter()
        e, si = obj.kernel()
        dt = time.perf_counter() - t0
        times.append(dt)
        energy = e[0]
    t_mean = np.mean(times)
    t_std  = np.std(times)
    print(f"  {label:30s}  E0={energy:.8f}  t={t_mean:.2f}±{t_std:.2f}s  ({N_REPEAT} runs)")
    return energy, t_mean


print("\n── Timing ──────────────────────────────────────────────────────────────")
e_opt,   t_opt   = run_and_time(LSI_LUSCC,       "With optimization (opt2)")
e_noopt, t_noopt = run_and_time(LSI_LUSCC_NoOpt, "Without optimization    ")

print(f"\nSpeedup:  {t_noopt/t_opt:.2f}×")
print(f"Energy difference: {abs(e_opt - e_noopt):.2e} Ha  (should be < 1e-10)")
