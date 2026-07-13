"""Run LAS-LUSCC for Fe4 from the saved deposited-configuration LAS."""

import argparse
import sys
from time import time

import numpy as np
from mrh.exploratory.citools import grad
from mrh.my_pyscf.mcscf.chkfile import load_las_

from common import DATA, build_deposited_las, build_mf, ensure_data
from helper.util import get_sorted_excitations
from lcc import LSI_LUSCC


def selection_label(frac, epsilon, smult):
    spin = "allspin" if smult is None else f"s{smult}"
    if epsilon is None:
        return f"{spin}_frac{frac:.4f}"
    return f"{spin}_eps{epsilon:.6g}".replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frac", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--lindep-thresh", type=float, default=1e-4)
    parser.add_argument("--norm-thresh", type=float, default=1e-12)
    parser.add_argument("--max-memory", type=int, default=400000)
    parser.add_argument("--smult", type=int, default=None)
    parser.add_argument("--nroots", type=int, default=None)
    args = parser.parse_args()

    ensure_data()
    chk_path = DATA / "fe4_las.chk"
    if not chk_path.exists():
        raise FileNotFoundError(f"Missing LAS checkpoint: {chk_path}")

    _, mf = build_mf("fe4_las_luscc.log", args.max_memory)
    las, _ = build_deposited_las(mf, args.max_memory)
    load_las_(las, str(chk_path))
    print(f"loaded LAS checkpoint {chk_path}", flush=True)
    print(f"LAS energy = {las.e_tot}", flush=True)

    t0 = time()
    _, g_sel, a_idxs_all, i_idxs_all = grad.get_grad_exact(las, epsilon=0.0)
    g_all = np.array(g_sel)[:, 0]
    print(f"get_grad_exact time: {time() - t0:.2f} s", flush=True)
    print(f"total excitations available: {len(a_idxs_all)}", flush=True)

    if args.epsilon is None:
        a_idxs, i_idxs, selected_grads = get_sorted_excitations(
            a_idxs_all, i_idxs_all, g_all, fraction=args.frac)
        print(f"selection fraction = {args.frac}", flush=True)
    else:
        a_idxs, i_idxs, selected_grads = get_sorted_excitations(
            a_idxs_all, i_idxs_all, g_all, epsilon=args.epsilon)
        print(f"selection epsilon = {args.epsilon}", flush=True)
    print(f"selected excitations: {len(a_idxs)}", flush=True)
    sys.stdout.flush()

    t0 = time()
    las_luscc = LSI_LUSCC(
        las,
        a_idxs,
        i_idxs,
        lindep_thresh=args.lindep_thresh,
        norm_thresh=args.norm_thresh,
    )
    if args.smult is not None:
        las_luscc.sisolver.smult = args.smult
        print(f"target spin multiplicity smult = {args.smult}", flush=True)
    if args.nroots is not None:
        las_luscc.sisolver.nroots = args.nroots
        print(f"target number of roots = {args.nroots}", flush=True)
    e_roots, si = las_luscc.kernel()
    elapsed = time() - t0
    si_arr = np.asarray(si)
    si0 = si_arr[:, 0]
    label = selection_label(args.frac, args.epsilon, args.smult)

    print(f"LAS-LUSCC time: {elapsed:.2f} s", flush=True)
    print(f"LAS-LUSCC ground state energy = {e_roots[0]}", flush=True)
    print(f"LAS-LUSCC number of basis states: {si_arr.shape[0]}", flush=True)
    print(f"LAS-LUSCC number of roots: {len(e_roots)}", flush=True)

    np.save(DATA / f"fe4_luscc_{label}_e.npy", np.asarray(e_roots))
    np.save(DATA / f"fe4_luscc_{label}_si0.npy", si0)
    np.save(DATA / f"fe4_luscc_{label}_grads_sel.npy", np.asarray(selected_grads))
    np.save(DATA / f"fe4_luscc_{label}_a_idxs.npy",
            np.array(a_idxs, dtype=object), allow_pickle=True)
    np.save(DATA / f"fe4_luscc_{label}_i_idxs.npy",
            np.array(i_idxs, dtype=object), allow_pickle=True)
    np.savez(
        DATA / f"fe4_luscc_{label}_results.npz",
        e_roots=np.asarray(e_roots),
        si=si_arr,
        elapsed=np.asarray(elapsed),
        frac=np.asarray(args.frac),
        epsilon=np.asarray(np.nan if args.epsilon is None else args.epsilon),
        smult=np.asarray(-1 if args.smult is None else args.smult),
        nroots_requested=np.asarray(-1 if args.nroots is None else args.nroots),
    )

    top = np.argsort(-np.abs(si0))[:10]
    print(f"Top-10 |coeff| basis indices: {top.tolist()}", flush=True)
    print(f"Top-10 coefficients: {si0[top].tolist()}", flush=True)


if __name__ == "__main__":
    main()
