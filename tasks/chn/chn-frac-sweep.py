import numpy as np
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.exploratory.citools import grad
from mrh.my_pyscf import lassi
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from lcc import LSI_LUSCC
from helper.util import get_sorted_excitations
from time import time

# ── 1. System parameters ───────────────────────────────────────────────────
FRACS = np.arange(0.4, 1.01, 0.1)

ncas_f         = (4, 2, 4)
nelecas_f      = ((2, 2), (1, 1), (2, 2))
spin_sub_f     = (1, 1, 1)
frag_atom_list = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]

# ── 2. Molecule + RHF ──────────────────────────────────────────────────────
lib.logger.TIMER_LEVEL = lib.logger.INFO
mol = struct(2.0, 2.0, '6-31g')
mol.output = 'c2h4n4_631g.log'
mol.verbose = 3
mol.spin = 8
mol.max_memory = 200000  # 200 GB
mol.build()
mf = scf.RHF(mol).run()

# ── 3. LASSCF ──────────────────────────────────────────────────────────────
las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f)
mo_coeff = las.localize_init_guess(frag_atom_list)
las.kernel(mo_coeff)
print("LASSCF energy =", las.e_tot)

# ── 4. LASSIS ──────────────────────────────────────────────────────────────
lsi = lassi.LASSIS(las)
e_lsi, si = lsi.kernel()
print("LASSIS energy =", e_lsi[0])

# ── 5. All excitations (compute once) ─────────────────────────────────────
_, g_sel, a_idxs_all, i_idxs_all = grad.get_grad_exact(las, epsilon=0.0)
g_all = np.array(g_sel)[:, 0]
a_idxs_sorted, i_idxs_sorted, _ = get_sorted_excitations(
    a_idxs_all, i_idxs_all, g_all, fraction=1.0)
n_total = len(a_idxs_sorted)
print(f"Total excitations: {n_total}")

# ── 6. Sweep over fractions ────────────────────────────────────────────────
print(f"\n{'Frac':>6}  {'N_exc':>6}  {'E_gs':>20}  {'Time(s)':>8}")
print("-" * 46)

results = []
for frac in FRACS:
    frac = round(frac, 2)
    a_idxs, i_idxs, _ = get_sorted_excitations(
        a_idxs_all, i_idxs_all, g_all, fraction=frac)
    t0 = time()
    lsi_luscc = LSI_LUSCC(las, a_idxs, i_idxs)
    e_roots, _ = lsi_luscc.kernel()
    elapsed = time() - t0
    e_gs = e_roots[0]
    print(f"{frac:>6.2f}  {len(a_idxs):>6d}  {e_gs:>20.10f}  {elapsed:>8.2f}")
    results.append((frac, len(a_idxs), e_gs, elapsed))

# ── 7. Save results ────────────────────────────────────────────────────────
results = np.array(results)
np.save('frac_sweep_results.npy', results)
print("\nResults saved to frac_sweep_results.npy")
print("Columns: fraction, n_excitations, e_gs, time_s")
