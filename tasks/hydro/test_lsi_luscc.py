"""Tests for the unified LASSI_LSCC class (lsi-luscc / las-luscc).

Verification strategy
---------------------
1. Backward-compat (las-luscc):
   LASSI_LSCC(las, ...) must give the same ground-state energy as before the
   refactor, confirmed against a known reference (CASCI FCI energy for H4).

2. Single-rootspace LASSI == bare LAS:
   When the LASSI object wraps a single-root LAS (trivial 1×1 SI), passing that
   LASSI object to LASSI_LSCC must give the same energy as passing the bare LAS.

3. State-selection threshold:
   For a multi-root LASSI where one component dominates (|c₀| ≈ 1), setting
   threshold=1.0 (nothing passes) must fall back to the dominant component only,
   and threshold=0.0 must include all components.
"""
import numpy as np
import pytest
from copy import deepcopy
from pathlib import Path
from pyscf import gto, scf, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi as lassi_mod
from mrh.exploratory.citools.grad import get_grad_exact, get_grad_exact_lassi
from lcc import LSI_LUSCC as LASSI_LSCC


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _build_h4(basis='sto-3g'):
    pwd = Path(__file__).resolve().parent
    geom_path = pwd.parent / 'geom' / 'h4.xyz'
    with geom_path.open('r') as f:
        xyz = f.read()
    mol = gto.M(atom=xyz, basis=basis, output='/dev/null', verbose=0)
    mf = scf.RHF(mol).run()
    return mol, mf


@pytest.fixture(scope='module')
def h4_las():
    """Single-root H4 LAS (ground state only)."""
    mol, mf = _build_h4()
    las = LASSCF(mf, (2, 2), (2, 2), spin_sub=(1, 1))
    mo_loc = las.localize_init_guess([(0, 1), (2, 3)], mf.mo_coeff)
    las.kernel(mo_loc)
    return las


@pytest.fixture(scope='module')
def h4_casci():
    """FCI reference energy for H4/STO-3G."""
    mol, mf = _build_h4()
    mc = mcscf.CASCI(mf, 4, 4).run()
    return mc.e_tot


@pytest.fixture(scope='module')
def h4_single_lsi(h4_las):
    """Single-rootspace LASSI wrapping the single-root LAS."""
    lsi = lassi_mod.LASSI(h4_las)
    lsi.kernel()
    return lsi


@pytest.fixture(scope='module')
def h4_multi_lsi(h4_las):
    """Two-rootspace LASSI (singlet+singlet and singlet+triplet)."""
    las2 = h4_las.state_average(
        [1, 0],
        smults=[[1, 1], [1, 3]],
        charges=[[0, 0], [0, 0]],
    )
    las2.lasci()
    lsi = lassi_mod.LASSI(las2)
    lsi.kernel()
    return lsi


def _select_excitations(las, epsilon=0.001):
    """Return sorted (a_idxs, i_idxs) by gradient magnitude."""
    _, g_sel, a_idxs, i_idxs = get_grad_exact(las, epsilon=epsilon)
    order = np.argsort(-np.abs(np.array(g_sel)[:, 0]))
    return [a_idxs[k] for k in order], [i_idxs[k] for k in order]


# ---------------------------------------------------------------------------
# Test 1: backward compatibility – bare LAS input
# ---------------------------------------------------------------------------

def test_las_luscc_backward_compat(h4_las, h4_casci):
    """LASSI_LSCC(las, ...) should recover close to FCI for H4."""
    a_idxs, i_idxs = _select_excitations(h4_las, epsilon=0.0)
    lsi_lscc = LASSI_LSCC(deepcopy(h4_las), a_idxs, i_idxs)
    e_roots, _ = lsi_lscc.kernel()
    # Energy should be below LAS and within 10 mHa of FCI
    assert e_roots[0] < h4_las.e_tot, "LUSCC energy should be below LAS"
    assert abs(e_roots[0] - h4_casci) < 0.01, (
        f"LUSCC energy {e_roots[0]:.6f} too far from FCI {h4_casci:.6f}")


# ---------------------------------------------------------------------------
# Test 2: single-rootspace LASSI == bare LAS
# ---------------------------------------------------------------------------

def test_single_lsi_matches_las(h4_las, h4_single_lsi):
    """LASSI_LSCC(lsi, ...) with trivial SI must match LASSI_LSCC(las, ...)."""
    a_idxs, i_idxs = _select_excitations(h4_las, epsilon=0.001)

    e_las, _ = LASSI_LSCC(deepcopy(h4_las), a_idxs, i_idxs).kernel()
    e_lsi, _ = LASSI_LSCC(h4_single_lsi, a_idxs, i_idxs, state=0).kernel()

    assert abs(e_las[0] - e_lsi[0]) < 1e-8, (
        f"LAS path {e_las[0]:.10f} vs LSI path {e_lsi[0]:.10f}")


# ---------------------------------------------------------------------------
# Test 3: threshold controls number of reference states
# ---------------------------------------------------------------------------

def test_threshold_fallback(h4_multi_lsi):
    """threshold=1.0 should fall back to dominant component (not crash)."""
    lsi = h4_multi_lsi
    _, g_sel, a_idxs, i_idxs = get_grad_exact_lassi(lsi, state=0)
    order = np.argsort(-np.abs(np.array(g_sel)[:, 0]))
    a_idxs = [a_idxs[k] for k in order]
    i_idxs = [i_idxs[k] for k in order]

    # threshold so high nothing passes → fallback to 1 dominant component
    lscc_tight = LASSI_LSCC(lsi, a_idxs, i_idxs, threshold=1.0)
    e_tight, _ = lscc_tight.kernel()

    # threshold=0.0 → all components included
    lscc_all = LASSI_LSCC(lsi, a_idxs, i_idxs, threshold=0.0)
    e_all, _ = lscc_all.kernel()

    # More reference states → more variational freedom → lower or equal energy
    assert e_all[0] <= e_tight[0] + 1e-8, (
        f"All-components energy {e_all[0]:.8f} should be <= "
        f"single-component energy {e_tight[0]:.8f}")
