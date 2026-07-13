"""Run CASCI for Fe3 using the converged LASSCF orbitals.

This uses the same molecule/basis/charge/spin and active-space electron count
as tasks/fe3/las_luscc.py, but replaces the LAS product wavefunction with a
full CASCI in the total CAS(15,16) space at the saved LASSCF orbitals.
"""

import argparse
from pathlib import Path
from time import time

import numpy as np
from pyscf import gto, scf, lib, mcscf


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
NCAS = 15
NELECAS = (13, 3)


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
        output=str(DATA / "casci_las_orbs.log"),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-memory", type=int, default=120000)
    args = parser.parse_args()

    las_mo_path = DATA / "las_15_16_is.npy"
    if not las_mo_path.exists():
        raise FileNotFoundError(f"Missing LASSCF orbital file: {las_mo_path}")
    las_mo = np.load(las_mo_path)
    print(f"loaded LASSCF orbitals from {las_mo_path}", flush=True)

    _, mf = build_mf(args.max_memory)

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

    np.save(DATA / "fe3_casci_las_orbs_ci.npy", np.asarray(ci))
    np.save(DATA / "fe3_casci_las_orbs_mo_energy.npy", np.asarray(mo_energy))
    np.savez(
        DATA / "fe3_casci_las_orbs_results.npz",
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
