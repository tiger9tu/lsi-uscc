#!/usr/bin/env python3
"""
Minimal demonstration of non‑orthogonal configuration‑interaction
recoupling with PyGNME + PySCF.

* Builds an H6 chain in STO‑3G
* Generates two multiconfigurational reference states
  – a CASSCF(4e,4o) and a CASCI(2e,2o)
* Couples them with the build_noci_matrices() helper
* Prints Hamiltonian, overlap, one‑RDMs, and recoupled energies
"""

import numpy as np
import scipy.linalg
from dataclasses import dataclass

from pyscf import gto, scf, mcscf, ao2mo
from pygnme import wick, utils

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def owndata(x: np.ndarray) -> np.ndarray:
    """Force NumPy array to own its data (PyGNME requirement)."""
    if not x.flags["OWNDATA"]:
        y = np.zeros_like(x)  # contiguous, owns memory
        y[:] = x
        x = y
    return x

@dataclass
class NOCIState:
    """Container holding everything needed for one reference state."""
    ci:    np.ndarray  # CAS-CI coefficient tensor (α_det, β_det)
    mo:    np.ndarray  # MO coefficients (nmo, nmo)
    nact:  int         # # active orbitals
    ncore: int         # # core orbitals

def build_noci_matrices(states, h1e, h2e, ovlp, nmo, nocc, enuc):
    """
    Assemble Hamiltonian H, overlap S, and state‑coupled 1‑RDMs for a list of
    non‑orthogonal multiconfigurational wave‑functions.  See previous answer
    for detailed docstring.
    """
    nstate = len(states)
    h  = np.zeros((nstate, nstate))
    s  = np.zeros((nstate, nstate))
    rdm1 = np.zeros((nstate, nstate, nmo, nmo))

    for x, stx in enumerate(states):
        vx = utils.fci_bitset_list(nocc - stx.ncore, stx.nact)
        refx = wick.reference_state[float](nmo, nmo, nocc,
                                           stx.nact, stx.ncore, stx.mo)

        for w in range(x, nstate):
            stw = states[w]
            vw  = utils.fci_bitset_list(nocc - stw.ncore, stw.nact)

            refw = wick.reference_state[float](nmo, nmo, nocc,
                                               stw.nact, stw.ncore, stw.mo)
            orbs = wick.wick_orbitals[float, float](refx, refw, ovlp)

            mb = wick.wick_rscf[float, float, float](orbs, enuc)
            mb.add_one_body(h1e)
            mb.add_two_body(h2e)

            for iwa, da in enumerate(vw):
                for iwb, db in enumerate(vw):
                    for ixa, ca in enumerate(vx):
                        for ixb, cb in enumerate(vx):
                            stmp, htmp = mb.evaluate(ca, cb, da, db)
                            coeff = stw.ci[iwa, iwb] * stx.ci[ixa, ixb]

                            h[x, w] += htmp * coeff
                            s[x, w] += stmp * coeff

                            tmpP1 = np.zeros((orbs.m_nmo, orbs.m_nmo))
                            mb.evaluate_rdm1(ca, cb, da, db, stmp, tmpP1)
                            rdm1[x, w] += tmpP1 * coeff

            # transform 1‑RDM back to canonical MO bases of |Ψ_w⟩ and |Ψ_x⟩
            rdm1[x, w] = np.linalg.multi_dot((stw.mo, rdm1[x, w], stx.mo.T))

            # hermitian symmetrisation
            h[w, x]    = h[x, w]
            s[w, x]    = s[x, w]
            rdm1[w, x] = rdm1[x, w].T

    return h, s, rdm1

# ------------------------------------------------------------
# main script
# ------------------------------------------------------------

def main():
    # ----- build test molecule -----
    mol = gto.Mole()
    mol.atom   = ";".join([f"H 0 0 {i}" for i in range(6)])  # linear H6
    mol.basis  = "sto-3g"
    mol.verbose = 0
    mol.build()

    # ----- RHF -----
    mf = scf.RHF(mol).run()

    h1e  = owndata(mf.get_hcore())
    h2e  = owndata(ao2mo.restore(1, mf._eri, mol.nao)
                   .reshape(mol.nao**2, mol.nao**2))
    ovlp = owndata(mf.get_ovlp())
    nmo  = mf.mo_coeff.shape[1]
    nocc = int(np.sum(mf.mo_occ > 0))

    # ----- multiconfigurational references -----
    # CASSCF(4e,4o)
    casscf1 = mcscf.CASSCF(mf, 4, 4)
    e1, _, ci1, mo1, _ = casscf1.kernel()
    ncas1, ncore1 = casscf1.ncas, casscf1.ncore

    print("CI1 : ", ci1)

    # CASCI(2e,2o)
    casci2 = mcscf.CASCI(mf, 2, 2)
    e2, _, ci2, mo2, _ = casci2.kernel()
    ncas2, ncore2 = casci2.ncas, casci2.ncore

    print("Reference energies:")
    print(f"  CASSCF(4e,4o): {e1:16.10f} Eh")
    print(f"  CASCI (2e,2o): {e2:16.10f} Eh\n")

    # ----- package into NOCIState objects -----
    states = [
        NOCIState(ci=owndata(ci1), mo=owndata(mo1),
                  nact=ncas1, ncore=ncore1),
        NOCIState(ci=owndata(ci2), mo=owndata(mo2),
                  nact=ncas2, ncore=ncore2),
    ]

    # ----- build NOCI matrices -----
    h, s, rdm1 = build_noci_matrices(states, h1e, h2e, ovlp,
                                     nmo, nocc, mol.energy_nuc())

    print("Hamiltonian matrix H:\n", h)
    print("\nOverlap matrix S:\n", s)

    # ----- generalised diagonalisation -----
    evals, evecs = scipy.linalg.eigh(h, b=s)

    print("\nRecoupled NOCI energies (Eh):")
    for i, e in enumerate(evals):
        print(f"  root {i}: {e:16.10f}")

    # (optional) display a 1‑RDM block
    print("\n⟨Ψ₀|γ|Ψ₀⟩ (first RDM1 block):\n", rdm1[0, 0])


if __name__ == "__main__":
    main()
