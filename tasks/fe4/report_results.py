"""Summarize Fe4 LAS, LAS-LUSCC, and CASCI result files."""

from pathlib import Path

import numpy as np


DATA = Path(__file__).resolve().parent / "data"


def scalar_from_npz(path, key):
    if not path.exists():
        return "missing"
    with np.load(path, allow_pickle=True) as data:
        value = data[key]
        return np.asarray(value).item() if np.asarray(value).shape == () else value


def main():
    print("Fe4 LAS/LAS-LUSCC/CASCI job summary")
    print(f"Result directory: {DATA}")

    las_npz = DATA / "fe4_las_results.npz"
    print(f"LAS result file: {las_npz if las_npz.exists() else 'missing'}")
    print(f"LAS energy: {scalar_from_npz(las_npz, 'e_tot')}")

    luscc_files = sorted(DATA.glob("fe4_luscc_*_results.npz"))
    if luscc_files:
        luscc_npz = luscc_files[-1]
        with np.load(luscc_npz, allow_pickle=True) as data:
            e_roots = np.asarray(data["e_roots"])
            elapsed = np.asarray(data["elapsed"]).item()
        print(f"LAS-LUSCC result file: {luscc_npz}")
        print(f"LAS-LUSCC ground-state energy: {e_roots[0]}")
        print(f"LAS-LUSCC roots: {len(e_roots)}")
        print(f"LAS-LUSCC elapsed seconds: {elapsed}")
    else:
        print("LAS-LUSCC result file: missing")

    casci_npz = DATA / "fe4_casci_las_orbs_results.npz"
    print(f"CASCI result file: {casci_npz if casci_npz.exists() else 'missing'}")
    print(f"CASCI total energy: {scalar_from_npz(casci_npz, 'e_tot')}")
    print(f"CASCI active energy: {scalar_from_npz(casci_npz, 'e_cas')}")
    print(f"CASCI spin square: {scalar_from_npz(casci_npz, 'spin_square')}")
    print(f"CASCI multiplicity: {scalar_from_npz(casci_npz, 'multiplicity')}")


if __name__ == "__main__":
    main()
