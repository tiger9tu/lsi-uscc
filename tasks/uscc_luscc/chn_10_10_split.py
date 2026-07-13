"""Run split CHN(10,10) LAS-LUSCC or LAS-USCC sweeps for FRAC=5-9%.

This is intentionally separate from chn_10_10.py, whose comparison workflow
writes combined LAS/USCC/LUSCC artifacts for each FRAC.  These split jobs write
method-specific files so the existing 1-4% comparison data are not overwritten.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from pyscf import mcscf
from mrh.exploratory.citools import fockspace, grad
from mrh.exploratory.unitary_cc import lasuccsd

from helper.util import get_sorted_excitations
from lcc import LSI_LUSCC
from tasks.uscc_luscc.chn_10_10 import build_las, DATA
from tasks.uscc_luscc.compare import energy_of, run_uscc, spin_square_of


FRACS = (0.05, 0.06, 0.07, 0.08, 0.09)


def _select_all(las):
    print("[setup] building exact LAS gradient excitation list", flush=True)
    _, g_sel_all, a_all, i_all = grad.get_grad_exact(las, epsilon=0.0)
    g_all = np.array(g_sel_all)[:, 0]
    print(f"[setup] total excitations available: {len(a_all)}", flush=True)
    return a_all, i_all, g_all


def run_luscc(las, a_all, i_all, g_all):
    for frac in FRACS:
        out_path = DATA / f"chn_10_10_luscc_frac{frac:.2f}.npz"
        if out_path.exists():
            print(f"[luscc frac={frac:.2f}] skip existing {out_path}", flush=True)
            continue

        print(f"\n[luscc frac={frac:.2f}] selecting operators", flush=True)
        a_sel, i_sel, g_sel = get_sorted_excitations(
            a_all, i_all, g_all, fraction=frac)
        print(f"[luscc frac={frac:.2f}] operators={len(a_sel)}", flush=True)

        t0 = time.time()
        solver = LSI_LUSCC(las, a_sel, i_sel)
        solver.verbose = 0
        e_roots, si = solver.kernel()
        dt = time.time() - t0

        np.savez(
            out_path,
            frac=frac,
            n_operators=len(a_sel),
            e_las=las.e_tot,
            e_luscc=float(e_roots[0]),
            e_roots=np.asarray(e_roots),
            si=np.asarray(si),
            nroots=solver.nroots,
            dt=dt,
            gradients=np.asarray(g_sel),
        )
        print(
            f"[luscc frac={frac:.2f}] E={float(e_roots[0]):.10f} "
            f"nroots={solver.nroots} dt={dt:.1f}s -> {out_path}",
            flush=True,
        )


def _h_eff(mf, las):
    mc_for_h = mcscf.CASCI(mf, las.ncas, las.nelecas)
    mc_for_h.mo_coeff = las.mo_coeff
    h1, ecore = mc_for_h.get_h1eff()
    h2 = mc_for_h.get_h2eff()
    return [ecore, h1, h2]


def _las_energy_check(mol, las, h_eff):
    dummy = lasuccsd.FCISolver_USCC(mol, [], [])
    dummy.norb_f = las.ncas_sub
    ci0_f = [
        np.squeeze(fockspace.hilbert2fock(ci[0], no, ne))
        for ci, no, ne in zip(las.ci, las.ncas_sub, las.nelecas_sub)
    ]
    psi = dummy.build_psi(ci0_f, las.ncas, las.ncas_sub, sum(las.nelecas), frozen=None)
    las_fci = psi.dp_ci(ci0_f)
    e_check = energy_of(psi, h_eff, las_fci)
    if abs(e_check - las.e_tot) > 1e-6:
        raise RuntimeError(
            f"LAS energy mismatch: rebuilt={e_check:.12f}, las={las.e_tot:.12f}")
    return e_check


def run_uscc_sweep(mol, mf, las, a_all, i_all, g_all):
    h_eff = _h_eff(mf, las)
    e_las_check = _las_energy_check(mol, las, h_eff)
    print(f"[uscc setup] <LAS|H|LAS>={e_las_check:.10f}", flush=True)

    for frac in FRACS:
        out_path = DATA / f"chn_10_10_uscc_frac{frac:.2f}.npz"
        if out_path.exists():
            print(f"[uscc frac={frac:.2f}] skip existing {out_path}", flush=True)
            continue

        print(f"\n[uscc frac={frac:.2f}] selecting operators", flush=True)
        a_sel, i_sel, g_sel = get_sorted_excitations(
            a_all, i_all, g_all, fraction=frac)
        print(f"[uscc frac={frac:.2f}] operators={len(a_sel)}", flush=True)

        t0 = time.time()
        e_uscc, psi_uscc = run_uscc(mol, mf, las, a_sel, i_sel)
        uscc_fci = psi_uscc.get_fcivec(psi_uscc.x)
        e_check = energy_of(psi_uscc, h_eff, uscc_fci)
        ss, mult = spin_square_of(psi_uscc, uscc_fci)
        dt = time.time() - t0
        if abs(e_check - e_uscc) > 1e-5:
            raise RuntimeError(
                f"USCC energy mismatch: solver={e_uscc:.12f}, rebuilt={e_check:.12f}")

        np.savez(
            out_path,
            frac=frac,
            n_operators=len(a_sel),
            e_las=las.e_tot,
            e_uscc=float(e_uscc),
            e_uscc_check=float(e_check),
            s2=float(ss),
            multip=float(mult),
            dt=dt,
            gradients=np.asarray(g_sel),
        )
        print(
            f"[uscc frac={frac:.2f}] E={float(e_uscc):.10f} "
            f"S2={float(ss):.6f} mult={float(mult):.3f} "
            f"dt={dt:.1f}s -> {out_path}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("luscc", "uscc"))
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    print(f"[setup] method={args.method} fracs={FRACS}", flush=True)
    mol, mf, las = build_las()
    print(f"[setup] RHF={mf.e_tot:.10f} LASSCF={las.e_tot:.10f}", flush=True)
    print(f"[setup] LAS root count used by LSI_LUSCC(las,...): 1", flush=True)

    a_all, i_all, g_all = _select_all(las)
    if args.method == "luscc":
        run_luscc(las, a_all, i_all, g_all)
    else:
        run_uscc_sweep(mol, mf, las, a_all, i_all, g_all)


if __name__ == "__main__":
    main()
