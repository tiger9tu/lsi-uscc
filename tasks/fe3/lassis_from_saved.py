"""Run Fe3 LASSIS from the saved LAS checkpoint/orbitals."""

import argparse
from pathlib import Path
from time import time

import numpy as np
from pyscf import gto, scf, lib
from mrh.my_pyscf.mcscf.chkfile import dump_las, load_las_
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi.lassis import LASSIS


PWD = Path(__file__).resolve().parent
DATA = PWD / "data"
GEOM = PWD / "fe3.xyz"

BASIS = {
    "C": "cc-pvdz",
    "H": "cc-pvdz",
    "O": "cc-pvtz",
    "Al": "cc-pvtz",
    "Fe": "cc-pvtz",
}
NCAS_F = (5, 5, 5)
NELECAS_F = ((5, 1), (4, 1), (4, 1))
SPIN_SUB = (5, 4, 4)


def read_xyz_atom(path):
    atoms = []
    for line in path.read_text().splitlines():
        fields = line.replace(",", " ").split()
        if len(fields) == 4:
            atoms.append(" ".join(fields))
    return "\n".join(atoms)


def build_mf(max_memory):
    DATA.mkdir(parents=True, exist_ok=True)
    lib.logger.TIMER_LEVEL = lib.logger.INFO

    mol = gto.M(
        atom=read_xyz_atom(GEOM),
        verbose=4,
        spin=10,
        charge=0,
        basis=BASIS,
        output=str(DATA / "fe3_lassis_from_saved.log"),
        max_memory=max_memory,
    )

    mf = scf.ROHF(mol)
    mf.init_guess = "atom"
    mf = mf.density_fit()
    mf.max_cycle = 1
    mf.kernel()
    mf = mf.newton()
    mf.max_cycle = 1

    hf_is = DATA / "hf_is.npy"
    if hf_is.exists():
        print(f"loading {hf_is}", flush=True)
        mf.mo_coeff = np.load(hf_is)
    mf.kernel()
    print(f"ROHF energy = {mf.e_tot}", flush=True)
    return mol, mf


def load_or_rebuild_las(mf, max_memory):
    chk_path = DATA / "fe3_las.chk"
    las = LASSCF(mf, NCAS_F, NELECAS_F, spin_sub=SPIN_SUB, verbose=4)
    las.max_memory = max_memory
    las.max_cycle_macro = 200
    las.chkfile = str(chk_path)

    if chk_path.exists():
        print(f"loading LAS checkpoint {chk_path}", flush=True)
        load_las_(las, str(chk_path))
        print(f"LASCI/LASSCF energy = {las.e_tot}", flush=True)
        return las

    las_mo_path = DATA / "las_15_16_is.npy"
    if not las_mo_path.exists():
        raise FileNotFoundError(f"Missing saved LASSCF orbitals: {las_mo_path}")
    las_mo = np.load(las_mo_path)
    print(f"loaded LASSCF orbitals from {las_mo_path}", flush=True)

    t0 = time()
    las.lasci(mo_coeff=las_mo)
    print(f"fixed-orbital LASCI energy = {las.e_tot}", flush=True)
    print(f"fixed-orbital LASCI time: {time() - t0:.2f} s", flush=True)
    dump_las(las, str(chk_path))
    print(f"stored LAS checkpoint {chk_path}", flush=True)
    return las


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-memory", type=int, default=120000)
    args = parser.parse_args()

    _, mf = build_mf(args.max_memory)
    las = load_or_rebuild_las(mf, args.max_memory)

    lsi = LASSIS(las)
    lsi.verbose = lib.logger.INFO
    lsi.max_memory = args.max_memory

    t0 = time()
    e_roots, si = lsi.kernel()
    elapsed = time() - t0
    si_arr = np.asarray(si)

    print(f"LASSIS ground state energy = {e_roots[0]}", flush=True)
    print(f"LASSIS number of roots = {len(e_roots)}", flush=True)
    print(f"LASSIS model space dim = {si_arr.shape[0]}", flush=True)
    print(f"LASSIS time: {elapsed:.2f} s", flush=True)

    np.save(DATA / "fe3_lassis_e.npy", np.asarray(e_roots))
    np.save(DATA / "fe3_lassis_si.npy", si_arr)
    np.savez(
        DATA / "fe3_lassis_results.npz",
        e_roots=np.asarray(e_roots),
        si=si_arr,
        elapsed=np.asarray(elapsed),
        nroots=np.asarray(len(e_roots)),
    )

    si0 = si_arr[:, 0]
    top = np.argsort(-np.abs(si0))[:10]
    print(f"Top-10 |coeff| basis indices: {top.tolist()}", flush=True)
    print(f"Top-10 coefficients: {si0[top].tolist()}", flush=True)


if __name__ == "__main__":
    main()
