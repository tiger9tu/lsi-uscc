"""Run Fe4 LASSI from the saved LAS checkpoint and time only LASSI."""

import argparse
from time import time

import numpy as np
from pyscf import lib
from mrh.my_pyscf.lassi.lassis import LASSIS
from mrh.my_pyscf.mcscf.chkfile import load_las_

from common import DATA, build_deposited_las, build_mf, ensure_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smult", type=int, required=True)
    parser.add_argument("--nroots", type=int, required=True)
    parser.add_argument("--max-memory", type=int, default=120000)
    args = parser.parse_args()

    ensure_data()
    chk_path = DATA / "fe4_las.chk"
    if not chk_path.exists():
        raise FileNotFoundError(f"Missing LAS checkpoint: {chk_path}")

    _, mf = build_mf(f"fe4_lassis_s{args.smult}.log", args.max_memory)
    las, _ = build_deposited_las(mf, args.max_memory)
    load_las_(las, str(chk_path))
    print(f"loaded LAS checkpoint {chk_path}", flush=True)
    print(f"LAS energy = {las.e_tot}", flush=True)

    lsi = LASSIS(las)
    lsi.verbose = lib.logger.INFO
    lsi.max_memory = args.max_memory
    lsi.cisolver_attr_spin_flips["max_cycle"] = 200
    lsi.max_cycle_macro = 100
    lsi.sisolver.smult = args.smult
    lsi.sisolver.nroots = args.nroots
    lsi.sisolver.max_cycle = 200
    lsi.sisolver.pspace_size = 400

    print(f"target spin multiplicity smult = {args.smult}", flush=True)
    print(f"target number of roots = {args.nroots}", flush=True)

    t0 = time()
    lsi.run(opt=1)
    elapsed = time() - t0
    e_roots = np.asarray(lsi.e_roots)
    si_arr = np.asarray(lsi.si)

    print(f"LASSI time excluding LAS setup = {elapsed:.2f} s", flush=True)
    print(f"LASSI ground state energy = {e_roots[0]}", flush=True)
    print(f"LASSI number of roots = {len(e_roots)}", flush=True)
    print(f"LASSI SI shape = {si_arr.shape}", flush=True)

    stem = f"fe4_lassis_s{args.smult}"
    np.save(DATA / f"{stem}_e.npy", np.asarray(e_roots))
    np.save(DATA / f"{stem}_si.npy", si_arr)
    np.savez(
        DATA / f"{stem}_results.npz",
        e_roots=np.asarray(e_roots),
        si=si_arr,
        elapsed_lassis=np.asarray(elapsed),
        smult=np.asarray(args.smult),
        nroots_requested=np.asarray(args.nroots),
        nroots=np.asarray(len(e_roots)),
    )

    si0 = si_arr[:, 0]
    top = np.argsort(-np.abs(si0))[:10]
    print(f"Top-10 |coeff| basis indices: {top.tolist()}", flush=True)
    print(f"Top-10 coefficients: {si0[top].tolist()}", flush=True)


if __name__ == "__main__":
    main()
