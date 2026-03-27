"""Tests for get_grad_exact_lassi.

Verification strategy:
  For a single-rootspace LASSI (trivial 1x1 SI matrix), the LASSI eigenstate
  is identical to the LAS ground state.  Therefore get_grad_exact_lassi must
  produce the same gradients as get_grad_exact.
"""
import numpy as np
import pytest
from pathlib import Path
from pyscf import gto, scf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi as lassi_mod
from mrh.exploratory.citools.grad import get_grad_exact, get_grad_exact_lassi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def h4_single_state():
    """H4 chain with STO-3G, single-rootspace LAS + trivial LASSI."""
    pwd = Path(__file__).resolve().parent
    geom_path = pwd.parent / 'geom' / 'h4.xyz'
    with geom_path.open('r') as f:
        xyz = f.read()

    mol = gto.M(atom=xyz, basis='sto-3g', output='/dev/null', verbose=0)
    mf = scf.RHF(mol).run()

    las = LASSCF(mf, (2, 2), (2, 2), spin_sub=(1, 1))
    mo_loc = las.localize_init_guess([(0, 1), (2, 3)], mf.mo_coeff)
    las.kernel(mo_loc)

    # Single-rootspace LASSI: SI matrix is 1x1 identity
    lsi = lassi_mod.LASSI(las)
    lsi.kernel()

    return las, lsi


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_grad_lassi_matches_las(h4_single_state):
    """Single-rootspace LASSI gradients must match LAS gradients exactly."""
    las, lsi = h4_single_state

    grads_las, _, _, _ = get_grad_exact(las)
    grads_lsi, _, _, _ = get_grad_exact_lassi(lsi, state=0)

    err = np.max(np.abs(grads_las - grads_lsi))
    assert err < 1e-10, f"max |grad_las - grad_lassi| = {err:.3e}"


def test_grad_lassi_selected_match(h4_single_state):
    """Selected gradients (epsilon=0) must be identical between LAS and LASSI."""
    las, lsi = h4_single_state

    _, g_sel_las, a_las, i_las = get_grad_exact(las)
    _, g_sel_lsi, a_lsi, i_lsi = get_grad_exact_lassi(lsi, state=0)

    assert len(g_sel_las) == len(g_sel_lsi), (
        f"Number of selected gradients differs: {len(g_sel_las)} vs {len(g_sel_lsi)}"
    )
    for (g_l, _), (g_i, _) in zip(g_sel_las, g_sel_lsi):
        assert abs(g_l - g_i) < 1e-10, f"Selected grad mismatch: {g_l} vs {g_i}"

    for al, ai in zip(a_las, a_lsi):
        assert list(al) == list(ai), f"a_idxs differ: {al} vs {ai}"
    for il, ii in zip(i_las, i_lsi):
        assert list(il) == list(ii), f"i_idxs differ: {il} vs {ii}"
