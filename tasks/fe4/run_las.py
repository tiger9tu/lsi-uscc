"""Run the deposited Fe4 22e/20o four-fragment LAS setup and save it."""

import argparse
from time import time

import numpy as np
from mrh.my_pyscf.mcscf.chkfile import dump_las
from mrh.my_pyscf.tools import molden as mrh_molden

from common import DATA, build_deposited_las, build_mf, ensure_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-memory", type=int, default=400000)
    args = parser.parse_args()

    ensure_data()
    _, mf = build_mf("fe4_las.log", args.max_memory)
    las, mo_coeff = build_deposited_las(mf, args.max_memory)

    t0 = time()
    las.lasci_(mo_coeff)
    elapsed = time() - t0

    print(f"LASSCF/LASCI energy = {las.e_tot}", flush=True)
    print(f"LASSCF/LASCI time: {elapsed:.2f} s", flush=True)

    np.save(DATA / "fe4_las_mo_coeff.npy", las.mo_coeff)
    np.savez(
        DATA / "fe4_las_results.npz",
        e_tot=np.asarray(las.e_tot),
        e_states=np.asarray(las.e_states),
        ncas_sub=np.asarray(las.ncas_sub),
        nelecas_sub=np.asarray(las.nelecas_sub),
        elapsed=np.asarray(elapsed),
    )
    dump_las(las, str(DATA / "fe4_las.chk"))
    mrh_molden.from_lasscf(las, str(DATA / "fe4_las.molden"))
    print(f"stored LAS checkpoint {DATA / 'fe4_las.chk'}", flush=True)


if __name__ == "__main__":
    main()
