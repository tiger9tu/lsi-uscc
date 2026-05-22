"""Unit tests for spin-separated 3-RDM of LASSI wavefunctions.

Tests:
  1. Plan A (full LASSI CI vector) == Plan B (state-pair summation) for all states.
  2. Shape checks for make_casdm3s.
  3. Single-root LASSI 3-RDM matches direct PySCF calculation on outer-product CI.
  4. Trace relation: contracting one index of Gamma3_aaa gives (Na-2)*Gamma2_aa.
"""
import numpy as np
import pytest
from pathlib import Path
from pyscf import gto, scf, fci
from pyscf.fci.direct_spin1 import civec_spinless_repr, make_rdm123s
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi as lassi_mod
from mrh.my_pyscf.lassi import op_o0
from mrh.my_pyscf.lassi.op_o0 import ci_outer_product


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def h4_system():
    """H4 chain with STO-3G, 2 fragments of 2 orbitals each."""
    pwd = Path(__file__).resolve().parent
    geom_path = pwd.parent / 'geom' / 'h4.xyz'
    with geom_path.open('r') as f:
        xyz = f.read()
    mol = gto.M(atom=xyz, basis='sto-3g', output='/dev/null', verbose=0)
    mf = scf.RHF(mol).run()
    las = LASSCF(mf, (2, 2), (2, 2), spin_sub=(1, 1))
    mo_loc = las.localize_init_guess([(0, 1), (2, 3)], mf.mo_coeff)
    las.kernel(mo_loc)
    return las


@pytest.fixture(scope='module')
def h4_lassi(h4_system):
    """Two-state LASSI on H4: singlet+singlet and singlet+triplet rootspaces."""
    las = h4_system
    las2 = las.state_average(
        [1, 0],
        smults=[[1, 1], [1, 3]],
        charges=[[0, 0], [0, 0]],
    )
    las2.lasci()
    lsi = lassi_mod.LASSI(las2)
    e_roots, si = lsi.kernel()
    return las2, lsi, si


# ---------------------------------------------------------------------------
# Test 1: Plan A == Plan B
# ---------------------------------------------------------------------------

def test_planA_eq_planB(h4_lassi):
    """Plan A (one combined CI vector) and Plan B (pair summation) must agree."""
    _, lsi, si = h4_lassi
    ci_fr = lsi.ci
    nelec_frs = lsi.get_nelec_frs()
    for ix in range(si.shape[1]):
        rdm3_A = op_o0.root_make_rdm3s_planA(lsi, ci_fr, nelec_frs, si, ix)
        rdm3_B = op_o0.root_make_rdm3s(lsi, ci_fr, nelec_frs, si, ix)
        err = np.max(np.abs(rdm3_A - rdm3_B))
        assert err < 1e-10, (
            f"State {ix}: Plan A vs Plan B max |diff| = {err:.3e}"
        )


# ---------------------------------------------------------------------------
# Test 2: Shape checks
# ---------------------------------------------------------------------------

def test_shape_all_states(h4_lassi):
    """make_casdm3s() returns the correct shape for all eigenstates."""
    _, lsi, si = h4_lassi
    ncas = lsi.ncas
    nroots_si = si.shape[1]
    rdm3s = lsi.make_casdm3s()
    assert rdm3s.shape == (nroots_si, 4, ncas, ncas, ncas, ncas, ncas, ncas), (
        f"Expected ({nroots_si}, 4, {ncas}^6), got {rdm3s.shape}"
    )


def test_shape_single_state(h4_lassi):
    """make_casdm3s(state=0) returns shape (4, ncas^6)."""
    _, lsi, _ = h4_lassi
    ncas = lsi.ncas
    rdm3s = lsi.make_casdm3s(state=0)
    assert rdm3s.shape == (4, ncas, ncas, ncas, ncas, ncas, ncas), (
        f"Expected (4, {ncas}^6), got {rdm3s.shape}"
    )


# ---------------------------------------------------------------------------
# Test 3: Single-root LASSI == direct PySCF make_rdm123s on outer-product CI
# ---------------------------------------------------------------------------

def test_single_root_matches_pyscf(h4_lassi):
    """For each LAS product state (identity SI), the LASSI 3-RDM must match
    the result of fci.direct_spin1.make_rdm123s applied to the outer-product CI."""
    _, lsi, si = h4_lassi
    ci_fr = lsi.ci
    nelec_frs = lsi.get_nelec_frs()
    norb = lsi.ncas

    # Build all outer-product CI vectors
    ci_r, nelec_r = ci_outer_product(ci_fr, lsi.ncas_sub, nelec_frs)
    nprods = len(nelec_r)

    # Identity SI: each LASSI state = one LAS product state
    si_eye = np.eye(nprods)

    for ix in range(min(3, nprods)):
        # Reference: PySCF make_rdm123s directly on the outer-product CI
        ci_las = ci_r[ix]
        nelec = nelec_r[ix]
        _, _, ref_dm3s = make_rdm123s(ci_las, norb, nelec)
        ref = np.stack(ref_dm3s, axis=0)  # (4, ncas, ...)

        # Plan B with identity SI
        rdm3_B = op_o0.root_make_rdm3s(lsi, ci_fr, nelec_frs, si_eye, ix)
        err = np.max(np.abs(rdm3_B - ref))
        assert err < 1e-9, (
            f"LAS state {ix}: Plan B vs PySCF make_rdm123s max |diff| = {err:.3e}"
        )

        # Plan A with identity SI
        rdm3_A = op_o0.root_make_rdm3s_planA(lsi, ci_fr, nelec_frs, si_eye, ix)
        err = np.max(np.abs(rdm3_A - ref))
        assert err < 1e-9, (
            f"LAS state {ix}: Plan A vs PySCF make_rdm123s max |diff| = {err:.3e}"
        )


# ---------------------------------------------------------------------------
# Test 4: Trace relation Gamma3_aaa -> (Na-2) * Gamma2_aa
# ---------------------------------------------------------------------------

def test_trace_relation(h4_lassi):
    """Contracting creation-3/annihilation-3 pair in Gamma3_aaa gives (Na-2)*Gamma2_aa.

    PySCF convention after reorder_dm123:
        rdm3[p, s_, q, t, r, u] = <p+_α q+_α r+_α  u_α t_α s_α>   (for aaa)
    Trace over the 3rd creator-annihilator pair (indices 4 and 5):
        sum_r rdm3_aaa[p, s_, q, t, r, r] = (Na-2) * <p+_α q+_α t_α s_α>
    The alpha-alpha 2-RDM from make_casdm12s satisfies:
        rdm2s[0, p, s_, 0, q, t] = <p+_α q+_α t_α s_α>
    so the comparison is:
        trace3[p, s_, q, t]  ==  (Na-2) * rdm2s[0, p, s_, 0, q, t]
    """
    _, lsi, si = h4_lassi
    nelec_frs = lsi.get_nelec_frs()
    ci_fr = lsi.ci
    ci_r, nelec_r_list = ci_outer_product(ci_fr, lsi.ncas_sub, nelec_frs)

    for state in range(si.shape[1]):
        rdm3s = lsi.make_casdm3s(state=state)   # (4, n, n, n, n, n, n)
        rdm1s, rdm2s = lsi.make_casdm12s(state=state)
        # rdm2s shape: (2, ncas, ncas, 2, ncas, ncas)
        # rdm2s[sa, p, q, sb, r, s] = <p+_sa r+_sb s_sb q_sa>

        # Number of alpha electrons: use dominant LAS component
        dominant = np.argmax(np.abs(si[:, state]))
        na, nb = nelec_r_list[dominant]
        if na < 2:
            continue  # trace relation requires Na >= 2

        # Trace over the 3rd creator-annihilator pair (indices 4,5 of rdm3_aaa)
        trace3 = np.einsum('psqtrr->psqt', rdm3s[0])   # shape (n,n,n,n)
        # aa 2-RDM: rdm2s[0,:,:,0,:,:][p,s_,q,t] = <p+_α q+_α t_α s_α>
        ref = (na - 2) * rdm2s[0, :, :, 0, :, :]       # shape (n,n,n,n)
        err = np.max(np.abs(trace3 - ref))
        assert err < 1e-7, (
            f"State {state}: trace(Gamma3_aaa) vs (Na-2)*Gamma2_aa "
            f"max |diff| = {err:.3e}"
        )


# ---------------------------------------------------------------------------
# Test 5: Weighted average
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test 6: _make_rdm3s_frag_pair vs _make_rdm3s_spinless_pair for transitions
# ---------------------------------------------------------------------------

def test_frag_pair_transition():
    """_make_rdm3s_frag_pair must match _make_rdm3s_spinless_pair for same-rootspace
    transition pairs (bra != ket within one rootspace, i.e. lroots > 1).
    This exercises the case where dm1 is non-symmetric (bra != ket).
    """
    from mrh.my_pyscf.lassi.op_o0 import (
        _make_rdm3s_frag_pair, _make_rdm3s_spinless_pair, addr_outer_product
    )
    from pyscf.fci import cistring

    # 2 fragments: f0 (2 orbs, 1a 1b), f1 (2 orbs, 1a 0b)
    n0, ne0 = 2, (1, 1)
    n1, ne1 = 2, (1, 0)
    norb = n0 + n1
    ne_global = (ne0[0] + ne1[0], ne0[1] + ne1[1])

    rng = np.random.default_rng(17)
    ci_f0_a = rng.standard_normal((2, 2))
    ci_f0_a /= np.linalg.norm(ci_f0_a)
    ci_f0_b = rng.standard_normal((2, 2))
    ci_f0_b -= np.dot(ci_f0_b.ravel(), ci_f0_a.ravel()) * ci_f0_a
    ci_f0_b /= np.linalg.norm(ci_f0_b)
    ci_f1_a = rng.standard_normal((2, 1))
    ci_f1_a /= np.linalg.norm(ci_f1_a)
    ci_f1_b = rng.standard_normal((2, 1))
    ci_f1_b -= np.dot(ci_f1_b.ravel(), ci_f1_a.ravel()) * ci_f1_a
    ci_f1_b /= np.linalg.norm(ci_f1_b)

    def make_outer(f0, f1):
        nd_a = cistring.num_strings(norb, ne_global[0])
        nd_b = cistring.num_strings(norb, ne_global[1])
        ad_a = addr_outer_product([n0, n1], [ne0[0], ne1[0]])
        ad_b = addr_outer_product([n0, n1], [ne0[1], ne1[1]])
        nda0 = cistring.num_strings(n0, ne0[0]); ndb0 = cistring.num_strings(n0, ne0[1])
        nda1 = cistring.num_strings(n1, ne1[0]); ndb1 = cistring.num_strings(n1, ne1[1])
        dp = np.multiply.outer(f1.reshape(1, nda1, ndb1), f0.reshape(1, nda0, ndb0))
        dp = dp.transpose(0, 3, 1, 4, 2, 5).reshape(1, nda1 * nda0, ndb1 * ndb0)[0]
        ci = np.zeros((nd_a, nd_b))
        ci[np.ix_(ad_a, ad_b)] = dp
        return ci

    ci_bra = make_outer(ci_f0_a, ci_f1_a)
    ci_ket = make_outer(ci_f0_b, ci_f1_b)

    ref = _make_rdm3s_spinless_pair(ci_bra, ci_ket, norb, ne_global, ne_global)
    result = _make_rdm3s_frag_pair(
        [ci_f0_a, ci_f1_a], [ci_f0_b, ci_f1_b], [n0, n1], [ne0, ne1]
    )

    for spin, name in enumerate(['aaa', 'aab', 'abb', 'bbb']):
        err = np.max(np.abs(result[spin] - ref[spin]))
        assert err < 1e-12, (
            f"Transition pair {name}: _make_rdm3s_frag_pair vs spinless max|diff| = {err:.3e}"
        )


def test_weighted_average(h4_lassi):
    """Weighted average of rdm3s == manual sum."""
    _, lsi, si = h4_lassi
    nstates = si.shape[1]
    weights = np.ones(nstates) / nstates

    avg = lsi.make_casdm3s(weights=weights)
    rdm3s_all = lsi.make_casdm3s()
    ref_avg = np.tensordot(weights, rdm3s_all, axes=((0,), (0,)))

    err = np.max(np.abs(avg - ref_avg))
    assert err < 1e-12, f"Weighted average mismatch: {err:.3e}"
