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

from pyscf import gto, scf, mcscf, ao2mo, fci
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
            print("noccw = ", nocc, " nactw = ", stw.nact, " ncorew = ", stw.ncore)
            refw = wick.reference_state[float](nmo, nmo, nocc, stw.nact, stw.ncore, stw.mo)
            orbs = wick.wick_orbitals[float, float](refx, refw, ovlp)

            # orbsa = wick.wick_orbitals[float, float](nmo, nmo, 3, stx.mo, stw.mo, ovlp)
            # orbsb = wick.wick_orbitals[float, float](nmo, nmo, 3, stx.mo, stw.mo, ovlp)

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
                            # mb.evaluate_rdm1(ca, cb, da, db, stmp, tmpP1)
                            # rdm1[x, w] += tmpP1 * coeff

            # transform 1‑RDM back to canonical MO bases of |Ψ_w⟩ and |Ψ_x⟩
            # rdm1[x, w] = np.linalg.multi_dot((stw.mo, rdm1[x, w], stx.mo.T))

            # hermitian symmetrisation
            h[w, x]    = h[x, w]
            s[w, x]    = s[x, w]
            # rdm1[w, x] = rdm1[x, w].T

    return h, s, rdm1

# ------------------------------------------------------------
# main script
# ------------------------------------------------------------

import pyscf
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad, lasci_ominus1
from pyscf.fci import cistring

def fock_ci_to_cas_ci(norbcas, neleacas, nelebcas, uscc_ci):
    ''' Convert Fock space CI to CASCI, FOCK CI is the tensor product of
    fragment CIs, of size 2^norb x 2^norb. Many elemnts of FOCK CI
    are 0 and corresponds to invalid determinants. CASCI is a 
    c(norb,nelec) x c(norb,nelec) matrix. '''
    comb_str_a = cistring.make_strings(range(norbcas), neleacas)
    comb_str_b = cistring.make_strings(range(norbcas), nelebcas)
    cas_ci = np.zeros((len(comb_str_a), len(comb_str_b)), dtype=uscc_ci.dtype)
    for i, bra in enumerate(comb_str_a):
        for j, ket in enumerate(comb_str_b):
            cas_ci[i,j] = uscc_ci[bra, ket]
    return cas_ci


def main():
    # ----- build test molecule -----
    # Initializing the molecule with RHF
    #===================================
    xyz = '''H 0.0 0.0 0.0;
                H 1.0 0.0 0.0;
                H 0.2 1.6 0.1;
                H 1.159166 1.3 -0.1'''
    mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log.py',
        verbose=0)
    mf = scf.RHF (mol).run ()
    ref = mcscf.CASSCF (mf, 4, 4).run () # = FCI


    # Running LASSCF
    #===================================
    las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
    las.verbose = 4
    frag_atom_list = ((0,1),(2,3))
    mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
    las.kernel (mo_loc)

    # ----- RHF -----
    mf = scf.RHF(mol).run()

    h1e  = owndata(mf.get_hcore())
    h2e  = owndata(ao2mo.restore(1, mf._eri, mol.nao)
                   .reshape(mol.nao**2, mol.nao**2))
    ovlp = owndata(mf.get_ovlp())
    nmo  = mf.mo_coeff.shape[1]
    print("nmo = ", nmo)
    nocc = int(np.sum(mf.mo_occ > 0))
    # nocc = 3  # 3 occupied orbitals for each spin

    print("mf.mo_occ = ", mf.mo_occ)

    all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(las, epsilon=0.0001)
    # print ("All gradients = ", all_g)
    # print ("Selected gradients = ", g_sel)

    excitations = []
    for a, i in zip(a_idxs_selected, i_idxs_selected):
        excitations.append((tuple(i), tuple(a[::-1])))

    # print ("Selected excitations = ", excitations)

    #Computing energy through the LAS-UCC kernel using selected excitations
    #==========================================================================================
    epsilon=0.001
    mc_uscc = mcscf.CASCI(mf, 4, 4)


    mc_uscc.mo_coeff = las.mo_coeff
    lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
    mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
    mc_uscc.fcisolver.norb_f = [2,2]
    mc_uscc.fcisolver.frozen = 'CI'
    e1, _, ci1, mo1, _ =mc_uscc.kernel()

    print("mc_uscc ci shape = ", ci1.shape)
    mc_cas_ci = fock_ci_to_cas_ci(nmo, 2,2, ci1)
    print("mc_cas_ci shape = ", mc_cas_ci.shape)

    # # ----- multiconfigurational references -----
    # # CASSCF(4e,4o)
    # casscf1 = mcscf.CASSCF(mf, 4, (2,2))
    # e1, _, ci1, mo1, _ = casscf1.kernel()
    # print("mo1 = ", mo1)
    # print("ci1 = ", ci1)
    # ncas1, ncore1 = casscf1.ncas, casscf1.ncore

    # # print("CI1 : ", ci1)

    # # CASCI(2e,2o)
    # casci2 = mcscf.CASCI(mf, 2, (1,1))
    # e2, _, ci2, mo2, _ = casci2.kernel()
    # print("CI2 : ", ci2)
    # ncas2, ncore2 = casci2.ncas, casci2.ncore

    # print("Reference energies:")
    # print(f"  CASSCF(4e,4o): {e1:16.10f} Eh")
    # print(f"  CASCI (2e,2o): {e2:16.10f} Eh\n")


    # ----- package into NOCIState objects -----
    states = [
        NOCIState(ci=owndata(mc_cas_ci), mo=owndata(mo1),
                  nact=4, ncore=0),
        NOCIState(ci=owndata(mc_cas_ci), mo=owndata(mo1),
                  nact=4, ncore=0),
    ]

    # ----- build NOCI matrices -----
    h, s, rdm1 = build_noci_matrices(states, h1e, h2e, ovlp,
                                     nmo, nocc, mol.energy_nuc())

    print("Hamiltonian matrix H:\n", h)
    print("\nOverlap matrix S:\n", s)

    # # ----- generalised diagonalisation -----
    # evals, evecs = scipy.linalg.eigh(h, b=s)

    # print("\nRecoupled NOCI energies (Eh):")
    # for i, e in enumerate(evals):
    #     print(f"  root {i}: {e:16.10f}")

    # # (optional) display a 1‑RDM block
    # print("\n⟨Ψ₀|γ|Ψ₀⟩ (first RDM1 block):\n", rdm1[0, 0])


if __name__ == "__main__":
    main()
