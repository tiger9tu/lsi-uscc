"""Tests for lsi-luscc on C2H4N4 (CHN).

The lsi-luscc mode feeds a post-kernel LASSI object into LSI_LUSCC.
All LAS states whose |SI coefficient| > threshold (default 0.01) are used
as reference states; excitations are applied to each of them.

Verification strategy
---------------------
1. lsi-luscc energy <= LASSIS energy
   Using significant LAS components as references and applying selected
   excitations must give a variational improvement over bare LASSIS.

2. Multiple reference states are selected
   For the CHN ground state, several LAS components have |si| > 0.01, so
   the number of sig_indices must be > 1.

3. Stricter threshold → fewer references → higher or equal energy
   threshold=1.0 (nothing passes → fallback to 1 dominant component) should
   give energy >= threshold=0.0 (all components).
"""
import numpy as np
import pytest
from pyscf import scf, mcscf
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSI
from mrh.my_pyscf import lassi as lassi_mod
from mrh.exploratory.citools.grad import get_grad_exact_lassi
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from lcc import LSI_LUSCC


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def chn_lsi():
    """C2H4N4 LASSCF + LASSIS with default geometry (dnn=2.0, 2.0)."""
    mol = struct(2.0, 2.0, '6-31g')
    mol.verbose = 0
    mol.spin = 8
    mol.build()
    mf = scf.RHF(mol).run()

    las = LASSCF(mf, (4, 2, 4), ((2, 2), (1, 1), (2, 2)), spin_sub=(1, 1, 1))
    mo_coeff = las.localize_init_guess([[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]])
    las.kernel(mo_coeff)

    lsi = lassi_mod.LASSIS(las)
    e_roots, si = lsi.kernel()
    return las, lsi, e_roots, si


def _select_excitations(lsi, frac=0.02):
    """Return top-|grad| fraction of excitations from the LASSI state."""
    _, g_sel, a_idxs, i_idxs = get_grad_exact_lassi(lsi, state=0)
    g = np.array(g_sel)[:, 0]
    order = np.argsort(-np.abs(g))
    n = max(1, int(len(a_idxs) * frac))
    return [a_idxs[order[k]] for k in range(n)], [i_idxs[order[k]] for k in range(n)]


# ---------------------------------------------------------------------------
# Test 1: lsi-luscc energy <= LASSIS energy
# ---------------------------------------------------------------------------

def test_lsi_luscc_below_lassis(chn_lsi):
    """LSI_LUSCC(lsi, ...) ground state must be at or below LASSIS."""
    las, lsi, e_lsi, si = chn_lsi
    a_idxs, i_idxs = _select_excitations(lsi, frac=0.02)

    lsi_luscc = LSI_LUSCC(lsi, a_idxs, i_idxs, state=0, threshold=0.01)
    e_roots, _ = lsi_luscc.kernel()

    assert e_roots[0] <= e_lsi[0] + 1e-8, (
        f"LSI-LUSCC energy {e_roots[0]:.8f} should be <= "
        f"LASSIS energy {e_lsi[0]:.8f}")


# ---------------------------------------------------------------------------
# Test 2: multiple reference states selected with threshold=0.01
# ---------------------------------------------------------------------------

def test_multiple_sig_indices(chn_lsi):
    """Ground state of CHN LASSIS should have more than one |si| > 0.01."""
    las, lsi, e_lsi, si = chn_lsi
    a_idxs, i_idxs = _select_excitations(lsi, frac=0.01)

    solver = LSI_LUSCC(lsi, a_idxs, i_idxs, state=0, threshold=0.01)
    sig = solver._select_sig_indices()
    assert len(sig) > 1, (
        f"Expected multiple significant components with threshold=0.01, "
        f"got {len(sig)}: si={lsi.si[:, 0]}")


# ---------------------------------------------------------------------------
# Test 3: stricter threshold gives higher or equal energy
# ---------------------------------------------------------------------------

def test_threshold_tightens_energy(chn_lsi):
    """threshold=1.0 (fallback to 1 dominant) should give energy >= threshold=0.0."""
    las, lsi, e_lsi, si = chn_lsi
    a_idxs, i_idxs = _select_excitations(lsi, frac=0.02)

    e_all, _ = LSI_LUSCC(lsi, a_idxs, i_idxs, state=0, threshold=0.0).kernel()
    e_tight, _ = LSI_LUSCC(lsi, a_idxs, i_idxs, state=0, threshold=1.0).kernel()

    assert e_all[0] <= e_tight[0] + 1e-8, (
        f"All-components energy {e_all[0]:.8f} should be <= "
        f"single-component energy {e_tight[0]:.8f}")
