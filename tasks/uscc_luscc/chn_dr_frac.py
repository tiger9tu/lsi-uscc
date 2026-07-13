"""LAS-USCC + LAS-LUSCC on bisdiazene C2H4N4 at one (dr, frac) point.

Usage:
    python tasks/uscc_luscc/chn_dr_frac.py --dr 1.0 --frac 0.02

Builds (mol, mf, las) at dnn1=dnn2=dr in 6-31g with the (10,10) active space
used by chn_10_10.py (fragments (4,2,4), spin_sub (1,1,1), mol.spin=8). Runs
LAS-USCC and LAS-LUSCC at the requested operator fraction (top-|gradient|
selection), reconstructs all three states in the same Fock-space basis,
computes pairwise overlaps and the LAS / LUSCC / USCC / P|USCC> geometry,
and writes a single npz with everything the plotting scripts need.

If the output file already exists the script exits without recomputing.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from pyscf import scf, lib, mcscf
from mrh.exploratory.citools import fockspace, grad
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.tests.lasscf.c2h4n4_struct import structure as struct

from helper.util import get_sorted_excitations
from lcc import LSI_LUSCC
from tasks.uscc_luscc.compare import (
    build_rootspace_fci_vectors,
    decompose_in_plane,
    energy_of,
    run_uscc,
    solve_luscc_in_fock_basis,
    spin_square_of,
)


PWD = Path(__file__).resolve().parent
DATA = PWD / "data"


def build_las(dr: float):
    lib.logger.TIMER_LEVEL = lib.logger.INFO
    mol = struct(dr, dr, "6-31g")
    mol.output = str(DATA / f"chn_dr{dr:.1f}.log")
    mol.verbose = 3
    mol.spin = 8
    mol.max_memory = 60_000  # MB
    mol.build()
    mf = scf.RHF(mol).run()

    ncas_f = (4, 2, 4)
    nelecas_f = ((2, 2), (1, 1), (2, 2))
    spin_sub = (1, 1, 1)
    frag_atom_list = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]

    las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub)
    mo_coeff = las.localize_init_guess(frag_atom_list)
    las.kernel(mo_coeff)
    return mol, mf, las


def _h_eff(mf, las):
    mc = mcscf.CASCI(mf, las.ncas, las.nelecas)
    mc.mo_coeff = las.mo_coeff
    h1, ecore = mc.get_h1eff()
    h2 = mc.get_h2eff()
    return [ecore, h1, h2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dr",   type=float, required=True)
    parser.add_argument("--frac", type=float, required=True)
    args = parser.parse_args()

    dr, frac = args.dr, args.frac
    DATA.mkdir(parents=True, exist_ok=True)
    out_path = DATA / f"chn_dr{dr:.1f}_frac{frac:.2f}.npz"
    if out_path.exists():
        print(f"[skip] {out_path} already exists", flush=True)
        return

    print(f"[setup] dr={dr} frac={frac} -> {out_path}", flush=True)

    t_setup = time.time()
    mol, mf, las = build_las(dr)
    print(f"[setup] RHF={mf.e_tot:.10f}  LASSCF={las.e_tot:.10f}  "
          f"(dt={time.time() - t_setup:.1f}s)", flush=True)
    print(f"[setup] ncas={las.ncas} ncas_sub={tuple(las.ncas_sub)} "
          f"nelecas={las.nelecas} 2S={las.nelecas[0] - las.nelecas[1]}",
          flush=True)

    h_eff = _h_eff(mf, las)

    # |LAS> as a Fock-space FCI vector, with a psi object we can re-use
    # to evaluate <X|H|X> and <X|S^2|X> for LUSCC too.
    dummy = lasuccsd.FCISolver_USCC(mol, [], [])
    dummy.norb_f = las.ncas_sub
    ci0_f_LAS = [np.squeeze(fockspace.hilbert2fock(ci[0], no, ne))
                 for ci, no, ne in zip(las.ci, las.ncas_sub, las.nelecas_sub)]
    psi_LAS = dummy.build_psi(ci0_f_LAS, las.ncas, las.ncas_sub,
                              sum(las.nelecas), frozen=None)
    LAS_fci = psi_LAS.dp_ci(ci0_f_LAS)
    e_las_check = energy_of(psi_LAS, h_eff, LAS_fci)
    ss_LAS, mult_LAS = spin_square_of(psi_LAS, LAS_fci)
    if abs(e_las_check - las.e_tot) > 1e-6:
        raise RuntimeError(
            f"LAS energy mismatch: rebuilt={e_las_check:.12f}, las={las.e_tot:.12f}")

    print("[setup] computing exact-gradient operator list", flush=True)
    _, g_sel_all, a_all, i_all = grad.get_grad_exact(las, epsilon=0.0)
    g_all = np.array(g_sel_all)[:, 0]
    print(f"[setup] total excitations available: {len(a_all)}", flush=True)
    a_sel, i_sel, g_sel = get_sorted_excitations(
        a_all, i_all, g_all, fraction=frac)
    n_ops = len(a_sel)
    print(f"[setup] selected n_operators={n_ops}", flush=True)

    # --- LAS-USCC ---
    t_uscc = time.time()
    e_uscc, psi_uscc = run_uscc(mol, mf, las, a_sel, i_sel)
    uscc_fci = psi_uscc.get_fcivec(psi_uscc.x)
    e_uscc_check = energy_of(psi_uscc, h_eff, uscc_fci)
    ss_USCC, mult_USCC = spin_square_of(psi_uscc, uscc_fci)
    dt_uscc = time.time() - t_uscc
    if abs(e_uscc_check - e_uscc) > 1e-5:
        raise RuntimeError(
            f"USCC energy mismatch: solver={e_uscc:.12f}, rebuilt={e_uscc_check:.12f}")
    print(f"[uscc] E={e_uscc:.10f}  S2={ss_USCC:.4f}  mult={mult_USCC:.3f}  "
          f"dt={dt_uscc:.1f}s", flush=True)

    # --- LAS-LUSCC ---
    t_luscc = time.time()
    lsi = LSI_LUSCC(las, a_sel, i_sel)
    lsi.verbose = 0
    e_roots_lassi, _ = lsi.kernel()
    e_luscc_lassi = float(e_roots_lassi[0])

    vecs = build_rootspace_fci_vectors(lsi, psi_LAS)
    eigvals, eigvecs = solve_luscc_in_fock_basis(vecs, psi_LAS, h_eff)
    e_luscc = float(eigvals[0])
    c0 = eigvecs[:, 0]
    luscc_fci = np.zeros_like(LAS_fci)
    for r, v in enumerate(vecs):
        luscc_fci += c0[r] * v
    if np.dot(LAS_fci.ravel(), luscc_fci.ravel()) < 0:
        luscc_fci = -luscc_fci
    if np.dot(LAS_fci.ravel(), uscc_fci.ravel()) < 0:
        uscc_fci = -uscc_fci
    e_luscc_check = energy_of(psi_LAS, h_eff, luscc_fci)
    ss_LUSCC, mult_LUSCC = spin_square_of(psi_LAS, luscc_fci)
    dt_luscc = time.time() - t_luscc
    if abs(e_luscc - e_luscc_lassi) > 1e-5:
        raise RuntimeError(
            f"LUSCC eigenvalue mismatch: refit={e_luscc:.12f}, "
            f"lassi={e_luscc_lassi:.12f}")
    if abs(e_luscc_check - e_luscc) > 1e-6:
        raise RuntimeError(
            f"LUSCC energy mismatch: refit={e_luscc:.12f}, "
            f"rebuilt={e_luscc_check:.12f}")
    print(f"[luscc] E={e_luscc:.10f}  S2={ss_LUSCC:.4f}  mult={mult_LUSCC:.3f}  "
          f"dt={dt_luscc:.1f}s", flush=True)

    # --- decomposition / overlaps ---
    coord = decompose_in_plane(LAS_fci, luscc_fci, uscc_fci)
    print("[geom] overlap (LAS, USCC, LUSCC):", flush=True)
    for row in coord["overlap"]:
        print("    " + "  ".join(f"{v:+.8f}" for v in row), flush=True)
    print(f"[geom] theta(LAS,LUSCC)={coord['theta_LUSCC_deg']:.4f} deg  "
          f"USCC out-of-plane angle={coord['theta_USCC_out_deg']:.4f} deg  "
          f"out amp={coord['out_of_plane_amp']:.3e}", flush=True)

    np.savez(
        out_path,
        dr=dr, frac=frac, n_operators=n_ops,
        e_las=las.e_tot,
        e_uscc=e_uscc, e_uscc_check=e_uscc_check,
        e_luscc=e_luscc, e_luscc_check=e_luscc_check,
        e_luscc_lassi=e_luscc_lassi,
        s2_las=ss_LAS, s2_uscc=ss_USCC, s2_luscc=ss_LUSCC,
        mult_las=mult_LAS, mult_uscc=mult_USCC, mult_luscc=mult_LUSCC,
        uscc_x=np.asarray(psi_uscc.x),
        a_sel=np.array(a_sel, dtype=object),
        i_sel=np.array(i_sel, dtype=object),
        gradients=np.asarray(g_sel),
        overlap=coord["overlap"],
        LAS_xyz=coord["LAS_xyz"], LUSCC_xyz=coord["LUSCC_xyz"],
        USCC_xyz=coord["USCC_xyz"], PUSCC_xyz=coord["PUSCC_xyz"],
        dt_uscc=dt_uscc, dt_luscc=dt_luscc,
    )
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
