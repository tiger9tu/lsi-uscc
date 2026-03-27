"""
Test: LASSI 3-RDM vs PySCF CASCI 3-RDM for H6.

Steps:
  1. RHF + LASSCF on H6 (3 fragments, 2 orbs/2 elec each).
  2. LAS state-average to get |las0> (ground) and |las1> (first excited).
  3. Form artificial LASSI state |lsi> = 1/sqrt(2)|las0> + 1/sqrt(2)|las1>.
  4. Compute 3-RDM via our implementation (root_make_rdm3s).
  5. Build the full CASCI CI vector for |lsi> by summing outer-product CIs.
  6. Compute 3-RDM via PySCF's make_rdm123s and compare.
"""
import numpy as np
from pyscf import gto, scf, fci, mcscf
from pyscf.fci.direct_spin1 import make_rdm123s
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi as lassi_mod
from mrh.my_pyscf.lassi import op_o0
from mrh.my_pyscf.lassi.op_o0 import ci_outer_product, root_make_rdm3s
from pathlib import Path

# ── 1. System ────────────────────────────────────────────────────────────────
pwd = Path(__file__).resolve().parent
xyz = (pwd.parent / 'geom' / 'h6.xyz').read_text()

mol = gto.M(atom=xyz, basis='sto-3g', output='/dev/null', verbose=0)
mf  = scf.RHF(mol).run()

ncas_f     = (2, 2, 2)
nelecas_f  = (2, 2, 2)
spin_sub_f = (1, 1, 1)
frags      = ((0, 1), (2, 3), (4, 5))

# ── 2. LASSCF ground state ────────────────────────────────────────────────────
las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f, verbose=0)
mo  = las.localize_init_guess(frags, mf.mo_coeff)
las.kernel(mo)
print(f"LASSCF energy: {las.e_tot:.10f}")

# ── 3. Two rootspaces both with total (na=3, nb=3) ───────────────────────────
# Rootspace 0: all singlets        → (1,1),(1,1),(1,1) → total (3,3)
# Rootspace 1: trip+trip+sing      → (2,0),(0,2),(1,1) → total (3,3)
# Both outer-product CIs have shape (C(6,3), C(6,3)) = (20, 20), so they
# can be combined into a single CI vector for the reference CASCI comparison.
las2 = las.state_average(
    weights=[0.5, 0.5],
    smults=[[1, 1, 1], [3, 3, 1]],
    spins=[[0, 0, 0], [2, -2, 0]],   # 2*Sz: frag0 Sz=+1, frag1 Sz=-1, frag2 Sz=0
    charges=[[0, 0, 0], [0, 0, 0]],
)
las2.lasci()
print(f"LASCI state energies: {las2.e_states}")

lsi  = lassi_mod.LASSI(las2)
e_roots, si = lsi.kernel()
print(f"LASSI energies: {e_roots}")
print(f"SI matrix shape: {si.shape}")

ci_fr      = lsi.ci
nelec_frs  = lsi.get_nelec_frs()
norb_f     = lsi.ncas_sub
norb       = lsi.ncas
nfrags     = len(norb_f)
nprods     = si.shape[0]
print(f"nprods={nprods}, nfrags={nfrags}, norb_f={norb_f}, norb={norb}")

# ── 4. Construct artificial LASSI vector |lsi> = 1/√2 |las0> + 1/√2 |las1> ──
w = 1.0 / np.sqrt(2.0)
si_art = np.zeros((nprods, 1))
si_art[:, 0] = w * si[:, 0] + w * si[:, 1]
print(f"\nArtificial SI vector (first 6 components): {si_art[:6, 0]}")
print(f"Norm of artificial SI: {np.linalg.norm(si_art[:, 0]):.8f}")

# ── 5. Our 3-RDM via root_make_rdm3s ─────────────────────────────────────────
rdm3s_ours = root_make_rdm3s(lsi, ci_fr, nelec_frs, si_art, ix=0)
print(f"\nOur rdm3s shape: {rdm3s_ours.shape}")

# ── 6. Reference 3-RDM via PySCF CASCI ───────────────────────────────────────
# Build the full CAS CI vector for |lsi> = sum_I si_art[I,0] * |prod_state_I>
from pyscf.fci.direct_spin1 import civec_spinless_repr
ci_r, nelec_r = ci_outer_product(ci_fr, norb_f, nelec_frs)
print(f"Product state nelec: {nelec_r}")

# Both rootspaces must have the same total (na, nb) for CI vectors to be addable
nelec_r_set = set(tuple(x) for x in nelec_r)
if len(nelec_r_set) > 1:
    print(f"WARNING: product states have different nelec: {nelec_r_set}")
nelec_cas = nelec_r[0]
print(f"CAS nelec: {nelec_cas}")

ndeta = ci_r[0].shape[0]
ndetb = ci_r[0].shape[1]
print(f"CI shape: ({ndeta}, {ndetb})")
ci_lsi = np.zeros((ndeta, ndetb))
for I in range(nprods):
    ci_lsi += si_art[I, 0] * ci_r[I]
norm = np.linalg.norm(ci_lsi)
print(f"CAS CI vector norm: {norm:.8f}")

# Reference A: spin-resolved make_rdm123s
_, _, rdm3s_ref_list = make_rdm123s(ci_lsi, norb, nelec_cas)
rdm3s_refA = np.stack(rdm3s_ref_list, axis=0)

# Reference B: spinless make_dm123 + reorder (same path as our implementation)
N = sum(nelec_cas)
ci_lsi_sl = civec_spinless_repr([ci_lsi], norb, [nelec_cas])
dm1r, dm2r, dm3r = fci.rdm.make_dm123('FCI3pdm_kern_sf', ci_lsi_sl, ci_lsi_sl, 2*norb, (N, 0))
_, _, dm3_sl = fci.rdm.reorder_dm123(dm1r, dm2r, dm3r)
rdm3s_refB = np.stack([
    dm3_sl[:norb, :norb, :norb, :norb, :norb, :norb],
    dm3_sl[:norb, :norb, :norb, :norb, norb:, norb:],
    dm3_sl[:norb, :norb, norb:, norb:, norb:, norb:],
    dm3_sl[norb:, norb:, norb:, norb:, norb:, norb:],
], axis=0)

# Also reference C: Plan A (root_make_rdm3s_planA)
from mrh.my_pyscf.lassi.op_o0 import root_make_rdm3s_planA
rdm3s_refC = root_make_rdm3s_planA(lsi, ci_fr, nelec_frs, si_art, ix=0)

print(f"\nRef A (make_rdm123s) vs Ref B (spinless) aab diff: "
      f"{np.max(np.abs(rdm3s_refA[1] - rdm3s_refB[1])):.3e}")
print(f"Ref B (spinless)    vs Ref C (Plan A)   aab diff: "
      f"{np.max(np.abs(rdm3s_refB[1] - rdm3s_refC[1])):.3e}")

# Use spinless reference (same convention as our implementation)
rdm3s_ref = rdm3s_refB
print(f"Using spinless reference (Ref B)")

# ── 7. Compare ───────────────────────────────────────────────────────────────
spin_names = ['aaa', 'aab', 'abb', 'bbb']
print("\n── Comparison: our vs reference 3-RDM ──")
print(f"{'spin':>6}  {'max|ref|':>12}  {'max|diff|':>12}  {'rel_err':>12}")
all_pass = True
for s, name in enumerate(spin_names):
    diff    = np.max(np.abs(rdm3s_ours[s] - rdm3s_ref[s]))
    ref_max = np.max(np.abs(rdm3s_ref[s]))
    rel     = diff / ref_max if ref_max > 1e-15 else 0.0
    status  = "PASS" if diff < 1e-8 else "FAIL"
    if diff >= 1e-8:
        all_pass = False
    print(f"  {name:>4}  {ref_max:>12.4e}  {diff:>12.4e}  {rel:>12.4e}  {status}")

print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
