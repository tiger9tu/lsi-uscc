from __future__ import annotations
import sys
import copy
import numpy as np
import h5py
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.my_pyscf.tools import molden
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
# from lasvqe.las_vqe import LASVQE
# from qiskit_qulacs.qulacs_estimator import QulacsEstimator


import numpy as np
from typing import Iterable, List, Any, Tuple, Dict, Optional, TypedDict
from dataclasses import dataclass, field
import functools, time, sys

# External deps
from pyscf import gto, scf, mcscf
from scipy.linalg import eigh

# mrh stack
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.citools import grad, fockspace

THRESHOLD_COND = 8e5
AMPLITUDE = 200
VERBOSE = 2

def print_matrix(obj, digits=3):
    try:
        arr = np.asarray(obj)
    except Exception:
        print(repr(obj))
        return

    if arr.ndim == 0:
        print(format(arr.item(), f'.{digits}g'))
        return
    if arr.ndim > 2:
        print(repr(obj))
        return

    def fmt(x):
        if isinstance(x, (str, bytes)):
            return str(x)
        try:
            return format(x, f'.{digits}g')
        except Exception:
            return str(x)

    if arr.ndim == 1:
        strs = [fmt(x) for x in arr]
        # Use individual widths (no vertical alignment needed for single row),
        # but keep a single space between columns
        row = '[ ' + ' '.join(s for s in strs) + ' ]'
        print(row)
        return

    # 2D case: compute string repr and column widths for alignment
    rows_str = [[fmt(x) for x in row] for row in arr]
    ncols = arr.shape[1]
    col_widths = [max(len(rows_str[r][c]) for r in range(arr.shape[0])) for c in range(ncols)]

    padded_rows = []
    for r in range(arr.shape[0]):
        padded = [rows_str[r][c].rjust(col_widths[c]) for c in range(ncols)]
        padded_rows.append('[ ' + ' '.join(padded) + ' ]')

    print('[\n ' + '\n '.join(padded_rows) + '\n]')


def print_list_matrix(obj, digits=3):
    try:
        arr = np.asarray(obj)
    except Exception:
        print(repr(obj))
        return

    # If it's a 2D array-like, delegate to print_matrix
    if arr.ndim <= 2:
        print_matrix(obj, digits)
        return

    # If 1D iterable, recurse on elements
    else:
        for el in obj:
            print_list_matrix(el, digits)
        return


    
def cilas2f(lasci, norb_f, nelec_f):
    """Convert LAS CI (per-fragment) to full Fock-space CI."""
    ci_f = []
    for i, ci in enumerate(lasci):
        ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
    return ci_f

def _get_Sij(psi_i, psi_j, h) -> complex:
    ucj, _ = psi_j.hc_x(psi_j.x, h)[1:3]
    uci, _ = psi_i.hc_x(psi_i.x, h)[1:3]
    return uci.ravel().conj().dot(ucj.ravel())


def _get_Sij_Hij(psi_i, psi_j, h) -> Tuple[complex, complex]:
    ucj, hucj = psi_j.hc_x(psi_j.x, h)[1:3]
    uci, huci = psi_i.hc_x(psi_i.x, h)[1:3]
    uci = uci.ravel()
    return uci.conj().dot(ucj.ravel()), uci.conj().dot(hucj.ravel())


def _get_S_H(sel_psis, h):
    S = np.zeros((len(sel_psis), len(sel_psis)), dtype=np.complex128)
    H = np.zeros_like(S)
    for i in range(len(sel_psis)):
        for j in range(len(sel_psis)):
            S[i, j], H[i, j] = _get_Sij_Hij(sel_psis[i], sel_psis[j], h)
    return S, H


def _psi_selection(trial_psi, h) -> List[int]:
    selected_indices: List[int] = []
    S_inc = np.zeros((0, 0), dtype=np.complex128)
    for idx, psi in enumerate(trial_psi):
        n_sel = len(selected_indices)
        S_new = np.zeros((n_sel + 1, n_sel + 1), dtype=np.complex128)
        if n_sel > 0:
            S_new[:n_sel, :n_sel] = S_inc
            for j, jidx in enumerate(selected_indices):
                S_new[n_sel, j] = _get_Sij(psi, trial_psi[jidx], h)
                S_new[j, n_sel] = _get_Sij(trial_psi[jidx], psi, h)
        S_new[n_sel, n_sel] = _get_Sij(psi, psi, h)
        cond = np.linalg.cond(S_new)
        if cond < THRESHOLD_COND:
            selected_indices.append(idx)
            S_inc = S_new
            if VERBOSE >= 4:
                print(f"Selected Psi ~ excitation {idx} (cond={cond:.2e})")
                print("Psi = \n")
                print_list_matrix(psi.get_fcivec(), digits=3)
        elif VERBOSE >= 3:
            print(f"Discarding Psi ~ excitation {idx-1} due to linear dependence (cond={cond:.2e})")
            print("S matrix's last row/col:")
            print_matrix(S_new[-1,:])
            print("Psi = \n")
            print_list_matrix(psi.get_fcivec(), digits=3)
    return selected_indices




lib.logger.TIMER_LEVEL = lib.logger.INFO
mol = struct (2.0, 2.0, '6-31g')
mol.output = 'c2h4n4_631g_las3_rscan.log'
mol.verbose = lib.logger.INFO
mol.spin = 8
mol.build ()
mf = scf.RHF (mol).run ()

nact = 10
nact_frag= (4, 2, 4)
nelec = (5,5)
nelec_frag = ((2,2),(1,1),(2,2))
spin_sub = (1,1,1)

las = LASSCF (mf, nact_frag, nelec_frag, spin_sub=spin_sub)
mo_coeff = las.localize_init_guess ([[0,1,2],[3,4,5,6],[7,8,9]])
las.kernel (mo_coeff)
mo_coeff = las.mo_coeff

mc = mcscf.CASCI (mf, 10, (5,5)).set (fcisolver=csf_solver(mol,smult=1))
mc.kernel (mo_coeff)

sys.stderr.flush ()
print ("LASSCF((4,4),(2,2),(4,4)) energy =", las.e_tot)
print ("CASCI(10,10) energy =", mc.e_tot, flush=True)

fmt_str = 'ROW: {:.1f} {:s} {:.9f} {:s} {:.9f}'
e_lasscf = las.e_states[0]
print (fmt_str.format (2.0, str(las.converged), e_lasscf, str(mc.converged), mc.e_tot))

def compute_lasuslcc(las, eps):
    all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(las, eps)
    
    gredients = np.array(g_sel)[:,0]
    sorted_indices = np.argsort(-np.abs(gredients))
    a_idxs_selected = [a_idxs_selected[i] for i in sorted_indices]
    i_idxs_selected = [i_idxs_selected[i] for i in sorted_indices]
    if VERBOSE >= 2:
        print("total excitation count:", len(all_g))
        print("selected excitation count:", len(g_sel))

    
    mc_uscc = mcscf.CASCI(mf, nact, nelec)
    mc_uscc.mo_coeff = las.mo_coeff
    mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
    mc_uscc.fcisolver.norb_f = nact_frag
    # You can subclass and override to change freezing behavior:

    ci0 = cilas2f(las.ci, nact_frag, nelec_frag)
    h1eff, e_core = mc_uscc.get_h1eff(mc_uscc.mo_coeff)
    h2eff = mc_uscc.get_h2eff()
    h = [e_core, h1eff, h2eff]

    nc = len(a_idxs_selected) + 1

    # Build a fresh FCISolver context to generate psi objects
    fci = mc_uscc.fcisolver

    las_ci0_f = cilas2f(las.ci, nact_frag, nelec_frag)
    tmp_mc_uscc = copy.deepcopy(mc_uscc)
    tmp_mc_uscc.kernel(ci0=ci0)
    if not tmp_mc_uscc.converged:
        print("Warning: initial CI did not converge")
    print("LAS-USCC-VQE energy:", tmp_mc_uscc.e_tot)
    print("optimized amplitudes:", tmp_mc_uscc.fcisolver.psi.x)
    norb = sum(nact_frag)
    norb_f = nact_frag

    psis = []
    min_ci_eng = +np.inf
    for i in range(nc):
        psi = getattr(fci, 'psi', fci.build_psi(las_ci0_f, norb, norb_f, nelec))
        psi.uop.linearize = True
        if i >=1:
            psi.x[psi.nconstr + i - 1] = float(AMPLITUDE)
        energy = psi.energy_tot(psi.x, h)
        min_ci_eng = min(min_ci_eng, float(energy))
        psis.append(psi)

    if VERBOSE >= 3:
        print("Psi0 = \n")
        print_list_matrix(psis[0].hc_x(psis[0].x, h)[1], digits=3)
    if VERBOSE >= 2:
        print(f"NOQI trial states (incl ref): {nc}; min single-state E: {min_ci_eng:.12f}")

    sel = _psi_selection(psis, h)
    sel_psis = [psis[i] for i in sel]
    if VERBOSE >= 2:
        print(f"Selected {len(sel)} / {nc} states after cond filter")

    # Build S/H and solve
    S, H = _get_S_H(sel_psis, h)
    if VERBOSE >= 1:
        print("Condition number of S:", np.linalg.cond(S))
    if VERBOSE >= 4:
        print("S:"); print_matrix(S)
        print("H:"); print_matrix(H)

    eigvals, eigvecs = eigh(H, S)
    e_noqi = float(np.min(eigvals.real))
    print(f"LAS-USCC-NOQI energy: {e_noqi:.17f}")
    print("State coefficients: ", eigvecs[:,0])
    # lasvqe = LASVQE(mf, las, f_orbs=(4,2,4), f_elec=(4,2,4), f_atom_list=[[0,1,2],[3,4,5,6],[7,8,9]],  spin_sub=(1,1,1), selected=True, epsilon=eps)
    # vqe_en, vqe_result = lasvqe.run(estimator=QulacsEstimator(), gate_counts=True, verbose=3)
	# print(f"LAS-USCC-VQE Energy: {vqe_en:.12f} Ha | Epsilon: {eps:.12f}")

    return

# Epsilons =  [0.0140047  0.00700483 0.0070026  0.00466867 0.00420959 0.00240517 0.00233465 0.00228203 0.00172649 0.00138026 0.00109903 0.00079211 0.00074442 0.00052791 0.00026381]


lsi = las.as_scanner ()
# for dr in np.arange (3.5, 3.4, -0.1):
dr = 3.5

mol1 = struct (dr, dr, '6-31g')
e_lasscf = lsi (mol1)
mc = mcscf.CASCI (lsi._scf, 10, (5,5)).set (fcisolver = csf_solver (mol, smult=1))
# mc.max_cycle = 100
mc.kernel (lsi.mo_coeff)
e_casci = mc.e_tot
compute_lasuslcc(lsi,0.0140047)
