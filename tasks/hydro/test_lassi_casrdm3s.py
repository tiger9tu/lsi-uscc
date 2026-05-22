"""Implement and verify make_casrdm3s for LASSI wavefunctions.

Implementation (per issues/lassi-casrdm3s.md):
  |lsi> = sum_i c_i |las_i>  where |las_i> is a product state of fragment CIs.

  1. For each LAS product state |las_i>, build the full CAS CI vector via
     outer product of fragment CI vectors.
  2. Form the LASSI CI vector: ci_lsi = sum_i c_i * ci_r[i].
  3. Call PySCF FCI API (make_rdm123s) to obtain the spin-separated 3-RDM.

Verification:
  For a single-root LAS (si = [[1.0]]), the LASSI eigenstate IS the LAS ground
  state, so the result must match las.make_casdm3s().
"""
import numpy as np
from pathlib import Path
from pyscf import gto, scf, lib
from pyscf.fci.direct_spin1 import make_rdm123s
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSI
from mrh.my_pyscf.lassi.op_o0 import ci_outer_product
from mrh.exploratory.citools.grad import make_casrdm123s_lassi


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

def make_casrdm3s_lassi(lsi, state=0):
    """Compute spin-separated 3-RDM for a LASSI eigenstate.

    Builds the LASSI CI vector from the FCI CI vectors of all LAS product
    states, then calls the PySCF FCI API.

    Args:
        lsi   : LASSI object (kernel already called)
        state : int, index of the LASSI eigenstate (default 0 = ground state)

    Returns:
        rdm3s : ndarray of shape (4, ncas, ncas, ncas, ncas, ncas, ncas)
                Spin components [aaa, aab, abb, bbb] in normal-ordered
                convention: rdm3[p,s,q,t,r,u] = <p+_a q+_b r+_c u_c t_b s_a>
    """
    ci_fr     = lsi.ci
    norb_f    = lsi.ncas_sub
    norb      = lsi.ncas
    nelec_frs = lsi.get_nelec_frs()
    si        = lsi.si

    # Step 1: outer-product FCI CI vector for every LAS product state
    ci_r, nelec_r = ci_outer_product(ci_fr, norb_f, nelec_frs)

    # Step 2: form LASSI eigenstate CI as weighted sum over all product states
    ci_lsi = sum(si[i, state] * ci_r[i] for i in range(len(ci_r)))

    # Step 3: spin-separated 3-RDM via PySCF FCI API
    _, _, dm3s = make_rdm123s(ci_lsi, norb, nelec_r[0])

    return np.stack(dm3s)


# ---------------------------------------------------------------------------
# System setup
# ---------------------------------------------------------------------------

pwd = Path(__file__).resolve().parent
geom_path = pwd.parent / 'geom' / 'h4.xyz'
with geom_path.open('r') as f:
    xyz = f.read()

mol = gto.M(atom=xyz, basis='sto-3g', output='/dev/null', verbose=0)
mf = scf.RHF(mol).run()

# Single-state LASSCF for converged MOs
las = LASSCF(mf, (2, 2), (2, 2), spin_sub=(1, 1), verbose=0)
mo_loc = las.localize_init_guess([(0, 1), (2, 3)], mf.mo_coeff)
las.kernel(mo_loc)

# ---------------------------------------------------------------------------
# Multi-state LASCI: two charge-transfer product states
#   State 0: frag0(0a,1b) x frag1(2a,1b)  ->  total (2a, 2b)
#   State 1: frag0(1a,0b) x frag1(1a,2b)  ->  total (2a, 2b)
# LASSI eigenstates are ±1/√2 combinations — genuine linear combination.
# ---------------------------------------------------------------------------

las_ms = las.state_average(
    [0.5, 0.5],
    smults=[[2, 2], [2, 2]],
    spins=[[-1, 1], [1, -1]],
    charges=[[1, -1], [1, -1]],
)
las_ms.lasci()

lsi = LASSI(las_ms, break_symmetry=True)
e_roots, si = lsi.kernel()

print(f"Number of LAS product states : {si.shape[0]}")
print(f"LASSI energies               : {e_roots}")
print(f"Ground-state SI coefficients : {si[:, 0]}")

assert np.all(np.abs(si[:, 0]) > 0.1), \
    f"FAIL: ground state is not a genuine linear combination: si[:,0] = {si[:,0]}"

# Compute 3-RDM for the LASSI ground state
rdm3s_gs = make_casrdm3s_lassi(lsi, state=0)
print(f"\nGround-state 3-RDM shape     : {rdm3s_gs.shape}")
print(f"fp(rdm3s_gs)                 : {lib.fp(rdm3s_gs):.10f}")

# ---------------------------------------------------------------------------
# Verification: single-root LASSI (si = [[1.0]]) must match las.make_casdm3s()
# ---------------------------------------------------------------------------

lsi_1 = LASSI(las)
lsi_1.kernel()

rdm3s_lassi = make_casrdm3s_lassi(lsi_1, state=0)
rdm3s_ref   = las.make_casdm3s()

max_err = np.max(np.abs(rdm3s_lassi - rdm3s_ref))
print(f"\nVerification (single LAS state, si = [[1.0]]):")
print(f"  fp(ref)   : {lib.fp(rdm3s_ref):.10f}")
print(f"  fp(lassi) : {lib.fp(rdm3s_lassi):.10f}")
print(f"  max |err| : {max_err:.3e}")

assert max_err < 1e-8, f"FAIL: max |err| = {max_err:.3e}"
print("PASS")

# ---------------------------------------------------------------------------
# Verification: rdm1 and rdm2 from make_casrdm123s_lassi match lsi.make_casdm12s
# ---------------------------------------------------------------------------

rdm1_123, rdm2_123, rdm3_123 = make_casrdm123s_lassi(lsi_1, state=0)
rdm1_api, rdm2_api_raw = lsi_1.make_casdm12s(state=0)
# convert make_casdm12s rdm2 (2,n,n,2,n,n) to (3,n,n,n,n) for comparison
rdm2_api = np.stack([rdm2_api_raw[0,:,:,0,:,:],
                     rdm2_api_raw[0,:,:,1,:,:],
                     rdm2_api_raw[1,:,:,1,:,:]])

err1 = np.max(np.abs(rdm1_123 - rdm1_api))
err2 = np.max(np.abs(rdm2_123 - rdm2_api))
print(f"\nVerification (rdm1/rdm2 from make_rdm123s vs lsi.make_casdm12s):")
print(f"  rdm1 max |err| : {err1:.3e}")
print(f"  rdm2 max |err| : {err2:.3e}")

assert err1 < 1e-8, f"FAIL rdm1: max |err| = {err1:.3e}"
assert err2 < 1e-8, f"FAIL rdm2: max |err| = {err2:.3e}"
print("PASS")

# ---------------------------------------------------------------------------
# Verification: rdm1/rdm2 from make_casrdm123s_lassi match lsi.make_casdm12s
# for the multi-state (genuine linear combination) case
# ---------------------------------------------------------------------------

rdm1_123, rdm2_123, _ = make_casrdm123s_lassi(lsi, state=0)
rdm1_api, rdm2_api_raw = lsi.make_casdm12s(state=0)
rdm2_api = np.stack([rdm2_api_raw[0,:,:,0,:,:],
                     rdm2_api_raw[0,:,:,1,:,:],
                     rdm2_api_raw[1,:,:,1,:,:]])

err1 = np.max(np.abs(rdm1_123 - rdm1_api))
err2 = np.max(np.abs(rdm2_123 - rdm2_api))
print(f"\nVerification (multi-state LASSI: rdm1/rdm2 from make_rdm123s vs lsi.make_casdm12s):")
print(f"  rdm1 max |err| : {err1:.3e}")
print(f"  rdm2 max |err| : {err2:.3e}")

assert err1 < 1e-8, f"FAIL rdm1: max |err| = {err1:.3e}"
assert err2 < 1e-8, f"FAIL rdm2: max |err| = {err2:.3e}"
print("PASS")
