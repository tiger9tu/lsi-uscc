"""Run full CASCI in the Fe4 22e/20o space using saved LAS orbitals."""

import argparse
from time import time

import numpy as np
from pyscf import mcscf

from common import DATA, build_mf, ensure_data


NCAS = 20
NELECAS = (11, 11)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-memory", type=int, default=400000)
    args = parser.parse_args()

    ensure_data()
    las_mo_path = DATA / "fe4_las_mo_coeff.npy"
    if not las_mo_path.exists():
        raise FileNotFoundError(f"Missing LAS orbital file: {las_mo_path}")
    las_mo = np.load(las_mo_path)
    print(f"loaded LASSCF/LASCI orbitals from {las_mo_path}", flush=True)

    _, mf = build_mf("fe4_casci_las_orbs.log", args.max_memory)
    mc = mcscf.CASCI(mf, NCAS, NELECAS)
    mc.max_memory = args.max_memory
    mc.verbose = 4

    t0 = time()
    e_tot, e_cas, ci, mo_coeff, mo_energy = mc.kernel(las_mo)
    elapsed = time() - t0
    ss, mult = mc.fcisolver.spin_square(ci, NCAS, NELECAS)

    print(f"CASCI total energy = {e_tot}", flush=True)
    print(f"CASCI active energy = {e_cas}", flush=True)
    print(f"CASCI <S^2> = {ss} multiplicity = {mult}", flush=True)
    print(f"CASCI time: {elapsed:.2f} s", flush=True)

    np.save(DATA / "fe4_casci_las_orbs_ci.npy", np.asarray(ci))
    np.save(DATA / "fe4_casci_las_orbs_mo_energy.npy", np.asarray(mo_energy))
    np.savez(
        DATA / "fe4_casci_las_orbs_results.npz",
        e_tot=np.asarray(e_tot),
        e_cas=np.asarray(e_cas),
        spin_square=np.asarray(ss),
        multiplicity=np.asarray(mult),
        ncas=np.asarray(NCAS),
        nelecas=np.asarray(NELECAS),
        elapsed=np.asarray(elapsed),
    )


if __name__ == "__main__":
    main()
