#!/usr/bin/env python
"""
mp2_ci_print.py

Run MP2, build the corresponding CISD CI vector,
and print out the full vector plus its unpacked blocks.
"""

import numpy as np
from pyscf import gto, scf, mp, ci

def main():
    # -------------------------------
    # 1) Build molecule & RHF
    # -------------------------------
    mol = gto.Mole()
    mol.atom  = "H 0 0 0;  F 0 0 0.9"
    mol.basis = "sto-3g"
    mol.spin  = 0
    mol.build()

    mf = scf.RHF(mol).run()

    # -------------------------------
    # 2) MP2
    # -------------------------------
    mp2 = mp.MP2(mf).run()
    print(f"MP2 total energy: {mp2.e_tot:.12f} Eh")
    t2 = mp2.t2  # shape (nocc,nocc,nvir,nvir)

    # -------------------------------
    # 3) Pack into CISD vector
    # -------------------------------
    nocc = t2.shape[0]
    nvir = t2.shape[2]

    c0 = 1.0                         # HF reference amplitude
    c1 = np.zeros((nocc, nvir))     # no singles in MP2
    c2 = t2.copy()                   # doubles = MP2 t2

    ci_vec = ci.cisd.amplitudes_to_cisdvec(c0, c1, c2)

    # -------------------------------
    # 4) Print the raw CI vector
    # -------------------------------
    print("\n=== RAW CI VECTOR ===")
    print(ci_vec)
    print(f"Length of CI vector: {ci_vec.size}")

    # -------------------------------
    # 5) Unpack and print blocks
    # -------------------------------
    c0_u, c1_u, c2_u = ci.cisd.vector_to_amplitudes(ci_vec, nocc, nvir)

    print("\n=== UNPACKED AMPLITUDES ===")
    print(f"Reference c0: {c0_u:.6f}")
    print(f"Singles block shape: {c1_u.shape}")
    print("Singles amplitudes (should all be zero):")
    print(c1_u)

    print(f"\nDoubles block shape: {c2_u.shape}")
    print("Sample nonzero t2 amplitudes (up to 8 decimals):")
    count = 0
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    val = c2_u[i, j, a, b]
                    if abs(val) > 1e-8:
                        print(f"  t2[{i},{j},{a},{b}] = {val:.8f}")
                        count += 1
                        if count >= 10:
                            break
                if count >= 10:
                    break
            if count >= 10:
                break
        if count >= 10:
            break
    if count == 0:
        print("  (all doubles are zero?)")

if __name__ == "__main__":
    main()
