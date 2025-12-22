#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full one-file script (updated):
1) Do LASSCF + LASSI[r,q] at the starting geometry (dr00, dr01)
2) Run a scanner over dr, and at each point:
   - run LASSI scanner
   - run CASCI reference
   - run LCC on top of the CURRENT LASSI state

Key change:
- run_lcc_on_lassi() no longer takes a_idxs_sel/i_idxs_sel (nor a_idx/i_idx).
- Inside run_lcc_on_lassi():
  - builds full excitation list (a_idxs_full/i_idxs_full)
  - tries to load gradients from grad_path; if missing, computes & saves
  - selects top-|grad| fraction excitations
  - runs FCISolver_CC with selected excitations
  - prints gradients to file (grad_path) (and optional stdout if verbose>1)
"""

import time
import ast
import sys
import numpy as np
from scipy import linalg
from itertools import product
from pathlib import Path

from pyscf import scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi
from mrh.my_pyscf.tools import molden

from mrh.exploratory.unitary_cc import lasuccsd
from helper import util
from lcc.lcc_solver import FCISolver_CC

from c2h4n4_struct import structure as struct


# ------------------------ helper: robustly get updated LASSCF from scanner ------------------------ #
def _get_las_from_scanner(scanner_obj):
    """
    Different scanner implementations expose the updated underlying object with different attribute names.
    We try a few common ones. If none found, we error with a helpful message.
    """
    for name in ("base", "las", "_las", "mf_las", "mc"):
        if hasattr(scanner_obj, name):
            obj = getattr(scanner_obj, name)
            # We want the LASSCF object, not the RHF
            if obj is not None and obj.__class__.__name__.lower().startswith("lasscf"):
                return obj
    # Some scanners store the underlying object in .base but it's not LASSCF; try nested guesses.
    if hasattr(scanner_obj, "base"):
        b = getattr(scanner_obj, "base")
        for name in ("las", "_las"):
            if hasattr(b, name):
                obj = getattr(b, name)
                if obj is not None and obj.__class__.__name__.lower().startswith("lasscf"):
                    return obj

    raise AttributeError(
        "Cannot locate the updated LASSCF object from lsi_scanner. "
        "Please print(dir(lsi_scanner)) once and adjust _get_las_from_scanner()."
    )


# ------------------------ core function: LCC on top of current LASSI state ------------------------ #
def run_lcc_on_lassi(
    mol,
    mf,
    las,
    lsi_obj,             # the CURRENT LASSI object (after it has been updated/kernel'ed by scanner)
    ncas_f,
    nelecas,
    grad_path,           # Path to gradient cache file for THIS geometry
    frac=0.01,           # keep top frac excitations by |grad|
    dx=1e-5,             # finite-diff step
    t=np.pi / 2,
    verbose=0,
):
    """
    Perform LCC (FCISolver_CC) on top of the CURRENT LASSI state.

    New behavior:
      - Build full excitation list internally
      - Load gradients from grad_path if exists, else compute+save
      - Select a_idxs_sel/i_idxs_sel based on top-|grad| fraction
      - Run LCC using selected excitations

    Returns:
        e_lcc (float)
        lccsi (np.ndarray)   # complex vector from solver
        si_gs (np.ndarray)   # real GS coefficient vector in psi basis
        grads (list[float])  # gradient magnitudes aligned with full excitation list
        top_indices (np.ndarray) # selected indices (from full list; excludes inserted |lsi>)
    """

    # ------------------------
    # 0) Full excitation list (a_idxs_full/i_idxs_full)
    # ------------------------
    uop = lasuccsd.gen_uccsd_op(las.ncas, las.ncas_sub)
    a_idxs_full = uop.a_idxs
    i_idxs_full = uop.i_idxs

    # ------------------------
    # 1) Build a USCC builder so we can construct LASUCCTrialState psi objects
    # ------------------------
    mc_uscc = mcscf.CASCI(mf, int(np.sum(ncas_f)), nelecas)
    mc_uscc.mo_coeff = las.mo_coeff

    fci_builder = lasuccsd.FCISolver_USCC(mol, a_idxs_full, i_idxs_full)
    fci_builder.norb_f = ncas_f
    mc_uscc.fcisolver = fci_builder

    # ------------------------
    # 2) Build product-state CI vectors in the full CAS FCI space (lasci_fs)
    # ------------------------
    las_ci_fs = []
    nroots = len(lsi_obj.ci[0])
    nfrag = len(lsi_obj.ci)

    for root in range(nroots):
        nelecs = []
        norbs = []
        frageigci = []
        for frag in range(nfrag):
            nelec_frag = lsi_obj.fciboxes[frag].fcisolvers[root].nelec
            nelecs.append(nelec_frag)
            norbs.append(lsi_obj.fciboxes[frag].fcisolvers[root].norb)

            eigci = lsi_obj.ci[frag][root]
            if len(eigci.shape) == 2:
                eigci = eigci[np.newaxis, :, :]
            frageigci.append(eigci)

        for prodci in product(*frageigci):
            las_ci_fs.append(util.cilas2f(prodci, norbs, nelecs))

    # ------------------------
    # 3) Build H,S in the psi basis and solve generalized eigenproblem to get current LASSI GS vector
    # ------------------------
    h1eff, e_core = mc_uscc.get_h1eff()
    h2eff = mc_uscc.get_h2eff()
    h = [e_core, h1eff, h2eff]

    las_psis = [
        fci_builder.build_psi(ci_f, int(np.sum(ncas_f)), ncas_f, nelecas)
        for ci_f in las_ci_fs
    ]
    n = len(las_psis)

    S = np.zeros((n, n), dtype=complex)
    H = np.zeros((n, n), dtype=complex)

    for i in range(n):
        lasi, hlasi = las_psis[i].hc_x(las_psis[i].x, h)[1:3]
        lasi, hlasi = lasi.ravel(), hlasi.ravel()
        for j in range(i, n):
            lasj, hlasj = las_psis[j].hc_x(las_psis[j].x, h)[1:3]
            lasj, hlasj = lasj.ravel(), hlasj.ravel()

            Sij = lasi.conj().dot(lasj)
            Hij = lasi.conj().dot(hlasj)
            S[i, j] = Sij
            H[i, j] = Hij
            if i != j:
                S[j, i] = np.conj(Sij)
                H[j, i] = np.conj(Hij)

    e_vals, e_vecs = linalg.eig(H, S)
    idx = e_vals.argsort()
    e_vals = e_vals[idx]
    e_vecs = e_vecs[:, idx]

    # GS vector in psi basis
    si_vec = e_vecs[:, 0]       # complex in general
    si_gs = si_vec.real         # keep your convention

    if verbose:
        print(f"  [debug] psi-basis GS energy = {e_vals[0].real:.12f} Ha")

    # ------------------------
    # 4) Load gradients if available, else compute & write to grad_path
    # ------------------------
    grad_path = Path(grad_path)
    grad_path.parent.mkdir(parents=True, exist_ok=True)

    grads = None
    try:
        with grad_path.open("r") as f:
            txt = f.read().strip()
        if txt:
            txt = txt.replace("np.float64(", "").replace(")", "")
            grads = ast.literal_eval(txt)
    except FileNotFoundError:
        grads = None
    except Exception:
        grads = None

    if grads is None:
        grads = []
        start = time.time()

        # gradient index k corresponds to excitation (a_idxs_full[k], i_idxs_full[k])
        # perturb each psi.x at (psi.nconstr + k)
        for k in range(len(a_idxs_full)):
            uilass = []
            huilass = []

            for j in range(n):
                psi = las_psis[j]
                psi.x[psi.nconstr + k] = dx
                c, uc, huc, uhuc, c_f = psi.hc_x(psi.x, h)
                uilass.append(uc.ravel())
                huilass.append(huc.ravel())
                psi.x[psi.nconstr + k] = 0.0

            uiclsi = sum(si_vec[j] * uilass[j] for j in range(n))
            huiclsi = sum(si_vec[j] * huilass[j] for j in range(n))

            e_dx = (uiclsi.conj().dot(huiclsi)) / (uiclsi.conj().dot(uiclsi)) - e_vals[0]
            grad = float(np.abs(e_dx.real / dx))
            grads.append(grad)

        end = time.time()

        # Save gradients to file (requested)
        with grad_path.open("w") as f:
            f.write(str(grads))

        if verbose:
            print(
                f"  [grad] computed {len(a_idxs_full)} gradients in {end - start:.2f} s"
            )
            print(f"  [grad] saved to: {grad_path}")

    if verbose > 1:
        print("  [grad] gradients:")
        print(grads)

    # ------------------------
    # 5) Select top fraction excitations using gradients
    # ------------------------
    ncc = int(np.ceil(frac * len(grads)))
    ncc = max(ncc, 1)
    top_indices = np.argsort(np.abs(grads))[-ncc:][::-1]

    a_idxs_sel = [a_idxs_full[i] for i in top_indices]
    i_idxs_sel = [i_idxs_full[i] for i in top_indices]

    # include the |lsi> itself
    a_idxs_sel.insert(0, np.array([0], dtype=np.uint8))
    i_idxs_sel.insert(0, np.array([0], dtype=np.uint8))

    if verbose:
        print(f"  [sel] frac={frac} -> selected top {ncc}/{len(grads)} excitations")
        if verbose > 1:
            print("  [sel] indices:", top_indices)

    # ------------------------
    # 6) Run CC solver on top of the LASSI GS vector
    # ------------------------
    mc_lcc = mcscf.CASCI(mf, int(np.sum(ncas_f)), nelecas)
    mc_lcc.mo_coeff = las.mo_coeff

    mc_lcc.fcisolver = FCISolver_CC(mol, a_idxs_sel, i_idxs_sel, t=t)
    mc_lcc.fcisolver.norb_f = ncas_f
    mc_lcc.fcisolver.ci0_fs = las_ci_fs
    mc_lcc.fcisolver.si = si_gs

    mc_lcc.kernel()
    return mc_lcc.e_tot, mc_lcc.fcisolver.lccsi, si_gs, grads, top_indices


# ------------------------ main ------------------------ #
def main():
    pwd = Path(__file__).resolve().parent

    VERBOSE = 1

    pes_step = -2.0
    # LASSI[r,q]
    r = 1
    q = 2

    # select top fraction of excitations (based on gradients)
    frac = 0.04
    dx = 1e-5

    # fragment CAS definition
    ncas_f = (3, 3)
    nelecas_f = ((2, 1), (1, 2))
    nelecas = tuple(sum(x) for x in zip(*nelecas_f))  # (nalpha, nbeta)

    # starting geometry
    dr00 = 2.0
    dr01 = 2.0

    # ------------------------ build start point ------------------------ #
    mol = struct(dr00, dr01, "6-31g")
    mol.output = pwd / "data" / "c2h4n4_lassirq_631g.log"
    mol.verbose = lib.logger.INFO
    mol.build()

    mf = scf.RHF(mol).run()

    las = LASSCF(mf, ncas_f, nelecas_f)
    las = las.state_average(
        [0.5, 0.5],
        spins=[[1, -1], [-1, 1]],
        smults=[[2, 2], [2, 2]],
        charges=[[0, 0], [0, 0]],
    )

    mo = las.sort_mo([16, 18, 22, 23, 24, 26])
    mo = las.localize_init_guess((list(range(5)), list(range(5, 10))), mo)
    las.kernel(mo)

    molden.from_lasscf(las, pwd / "data" / "c2h4n4_lasscf66_631g.molden")

    mc_ref = mcscf.CASCI(mf, 6, 6).set(fcisolver=csf_solver(mol, smult=1))
    mc_ref.kernel(las.mo_coeff)
    molden.from_mcscf(
        mc_ref, pwd / "data" / "c2h4n4_casscf66_631g.molden", cas_natorb=True
    )

    print("LASSCF((3,3),(3,3)) energy =", las.e_tot)
    print("CASCI(6,6) energy =", mc_ref.e_tot)

    lsi = lassi.LASSIrq(las, r=r, q=q)
    e_roots, si_rq = lsi.kernel()

    print(f"LASSI[{r},{q}] energy =", e_roots[0])
    molden.from_lassi(las, pwd / "data" / "c2h4n4_lassirq_631g.molden", si=si_rq)

    if VERBOSE > 1:
        print(f"SI vector (LASSI[{r},{q}]):")
        print(si_rq[:, 0])

    # ------------------------ optional: run LCC once at start point ------------------------ #
    grad_path0 = pwd / "data" / f"lcc_grad_cas6r1q2_dnn{dr00:.2f}_{dr01:.2f}.txt"

    e_lcc0, lccsi0, si_gs0, grads0, top_idx0 = run_lcc_on_lassi(
        mol=mol,
        mf=mf,
        las=las,
        lsi_obj=lsi,
        ncas_f=ncas_f,
        nelecas=nelecas,
        grad_path=grad_path0,
        frac=frac,
        dx=dx,
        t=np.pi / 2,
        verbose=0,
    )
    print(f"LSILCCSD energy at start: {e_lcc0:.9f}")
    if VERBOSE:
        print("lccsi vector (start):")
        print(lccsi0.real)

    # ------------------------ scanner ------------------------ #
    lsi_scanner = lsi.as_scanner()

    for dr in np.arange(3.5, -0.31, pes_step):
        mol1 = struct(dr, dr, "6-31g")

        # LASSI scanner step (updates internal state)
        e_lassi = lsi_scanner(mol1)

        # CASCI reference (use mol1)
        mc = mcscf.CASCI(lsi_scanner._las._scf, int(np.sum(ncas_f)), nelecas).set(
            fcisolver=csf_solver(mol1, smult=1)
        )
        mc.max_cycle = 100
        mc.kernel(lsi_scanner.mo_coeff)
        e_casci = mc.e_tot

        # LCC on top of current LASSI state
        las_now = _get_las_from_scanner(lsi_scanner)

        # Per-geometry gradient cache file
        grad_path = pwd / "data" / f"lcc_grad_cas6r1q2_dnn{dr:.2f}_{dr:.2f}.txt"

        t0 = time.time()
        e_lcc, lccsi, si_gs, grads, top_idx = run_lcc_on_lassi(
            mol=mol1,
            mf=lsi_scanner._las._scf,
            las=las_now,
            lsi_obj=lsi_scanner,   # scanner carries updated .ci/.fciboxes in most implementations
            ncas_f=ncas_f,
            nelecas=nelecas,
            grad_path=grad_path,
            frac=frac,
            dx=dx,
            t=np.pi / 2,
            verbose=0,
        )
        t1 = time.time()

        print(
            f"Distance: {dr:.1f} | "
            f"LASSI Energy: {e_lassi:.9f} Ha | "
            f"CASCI Energy: {e_casci:.9f} Ha | "
            f"LSILCCSD Energy: {e_lcc:.9f} Ha | "
            f"LCC time: {t1 - t0:.2f}s"
        )

        if VERBOSE > 2:
            print("  lccsi.real =", lccsi.real)


if __name__ == "__main__":
    main()
