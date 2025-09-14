#!/usr/bin/env python3

import numpy as np
import scipy.linalg
from dataclasses import dataclass

from pyscf import gto, scf, mcscf, ao2mo, fci
from pygnme import wick, utils
from typing import Iterable
from mrh.exploratory.citools import fockspace


TRACE_PROGRESS = False  # Set to False to disable progress output
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
    ci:    any
    mo:    any
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

            # Generate list of bitsets for all configurations of electrons in orbitals
            # Here we assume it's RHF, so all the orbitals are doubly occupied
            #     \param nelec Total number of (doubly) occupied orbitals in the active space
            #     \param norb  Total number of spatial molecular orbitals in the active space
            # std::vector<bitset> fci_bitset_list(size_t nelec, size_t norb);

            vw  = utils.fci_bitset_list(nocc - stw.ncore, stw.nact)
            print("noccw = ", nocc, " nactw = ", stw.nact, " ncorew = ", stw.ncore)
            refw = wick.reference_state[float](nmo, nmo, nocc, stw.nact, stw.ncore, stw.mo)
            orbs = wick.wick_orbitals[float, float](refx, refw, ovlp)

            # orbsa = wick.wick_orbitals[float, float](nmo, nmo, 3, stx.mo, stw.mo, ovlp)
            # orbsb = wick.wick_orbitals[float, float](nmo, nmo, 3, stx.mo, stw.mo, ovlp)

            mb = wick.wick_rscf[float, float, float](orbs, enuc)
            mb.add_one_body(h1e)
            mb.add_two_body(h2e)

            total_iterations = len(vx) * len(vw) * len(vx) * len(vw)
            if TRACE_PROGRESS:
                print(f"Evaluating {total_iterations} cycles...")

            iter_count = 0

            for iwa, da in enumerate(vw):
                for iwb, db in enumerate(vw):
                    for ixa, ca in enumerate(vx):
                        for ixb, cb in enumerate(vx):

                            iter_count += 1
                            if TRACE_PROGRESS and iter_count % 100 == 0:
                                print(f"Progress: {iter_count}/{total_iterations} ({iter_count / total_iterations * 100:.2f}%)")
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

def fock_ci_to_cas_ci(ncas, neleacas, nelebcas, uscc_ci):
    ''' Convert Fock space CI to CASCI, FOCK CI is the tensor product of
    fragment CIs, of size 2^norb x 2^norb. Many elemnts of FOCK CI
    are 0 and corresponds to invalid determinants. CASCI is a 
    c(norb,nelec) x c(norb,nelec) matrix. '''
    comb_str_a = cistring.make_strings(range(ncas), neleacas)
    comb_str_b = cistring.make_strings(range(ncas), nelebcas)
    cas_ci = np.zeros((len(comb_str_a), len(comb_str_b)), dtype=uscc_ci.dtype)
    for i, bra in enumerate(comb_str_a):
        for j, ket in enumerate(comb_str_b):
            cas_ci[i,j] = uscc_ci[bra, ket]
    return cas_ci

def flatten(seq: Iterable) -> list:
    result = []
    for item in seq:
        if isinstance(item, (list, tuple, np.ndarray)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def cilas2f(lasci, norb_f, nelec_f):
    ''' nelec (na, nb)'''
    ci_f = []
    for i, ci in enumerate(lasci):
        ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
    return ci_f



def build_las_states(mf, frag_confs):
    """Build LASSCFState objects from molecule and fragment configurations."""
    states = []
    for i, frag_conf in enumerate(frag_confs):
        ncas = frag_conf['ncas']
        nelecas = frag_conf['nelecas']
        spin_sub = frag_conf['spin_sub']
        frag_atoms = frag_conf['frag_atom_list']

        las = LASSCF(mf, ncas, nelecas, spin_sub=spin_sub)
        mo_loc = las.localize_init_guess (frag_atoms, mf.mo_coeff)
        las.kernel (mo_loc)
        states.append(las)

    return states

def las2noci_state(mol, las):
    """Convert a list of LASSCF states to NOCIState objects."""
    
    # mc_uscc = mcscf.CASCI(mf, sum(flatten(las.ncas_sub)), sum(flatten(las.nelecas_sub)))
    # mc_uscc.mo_coeff = las.mo_coeff

    fci = lasuccsd.FCISolver_USCC(mol, [],[])
    ncas = sum(flatten(las.ncas_sub))
    nelec = sum(flatten(las.nelecas_sub))

    las_fock_ci = cilas2f(las.ci, las.ncas_sub, las.nelecas_sub)
    psi =  getattr (fci, 'psi', fci.build_psi (las_fock_ci, ncas, las.ncas_sub, nelec)) # not sure about this

    cas_fock_ci = psi.dp_ci(las_fock_ci)
    # Here I assume spin up = spin down, so ncas is the same for both
    neleca = nelecb = nelec // 2
    if nelec % 2 != 0:
        raise ValueError("nelec must be even for NOCIState conversion")
    cas_ci = fock_ci_to_cas_ci(ncas, neleca, nelecb, cas_fock_ci)

    print("mol.nao = ", mol.nao, " ncas = ", ncas, " nelec = ", nelec)

    return NOCIState(owndata(cas_ci), owndata(las.mo_coeff), ncas, las.ncore) # ncore 

data_dir = "/home/jinx/repo/qchem/las_uccsd_data"




def main():

    with open(data_dir + '/circle/H10.xyz', 'r', encoding='utf-8') as f:
        h10_circle_xyz = f.read()

    with open(data_dir + '/polyenes/geometries/c4.xyz', 'r', encoding='utf-8') as f:
        c4xyz = f.read()

    h4_xyz =   """H      0.000000000000   0.000000000000   0.000000000000
            H      1.000000000000   0.000000000000   0.000000000000
            H      0.273746762116   2.195450598147   0.100000000000
            H      1.232912762116   1.895450598147  -0.100000000000
            """
    h6_xyz = """H      0.000000000000   0.000000000000   0.000000000000
            H      1.000000000000   0.000000000000   0.000000000000
            H      0.273746762116   2.195450598147   0.100000000000
            H      1.232912762116   1.895450598147  -0.100000000000
            H      0.507178110854   4.193780995243   0.049334760036
            H      1.506140937609   3.988021397347  -0.049334760036"""

    h8_xyz = """H      0.000000000000   0.000000000000   0.000000000000
            H      1.000000000000   0.000000000000   0.000000000000
            H      0.273746762116   2.195450598147   0.100000000000
            H      1.232912762116   1.895450598147  -0.100000000000
            H      0.507178110854   4.193780995243   0.049334760036
            H      1.506140937609   3.988021397347  -0.049334760036
            H      0.845946518048   6.364231296231   0.197836732111
            H      1.674032054647   5.908472292654  -0.197836732111
        """

    h4_sto3g = {
        'name': 'H4_STO3G',
        'xyz': h4_xyz,
        'basis': 'sto-3g',
    }

    h4_frag1 = {
        'ncas': [2, 2],
        'nelecas': [2, 2],
        'spin_sub': [1, 1],
        'frag_atom_list': [[0, 1], [2, 3]]
    }

    h4_frag2 = {
        'ncas' : [4],
        'nelecas' : [4],
        'spin_sub' : [1],
        'frag_atom_list' : [[0, 1, 2, 3]]
    }

    h6_sto3g = {
        'name': 'H6_STO3G',
        'xyz': h6_xyz,
        'basis': 'sto-3g',
    }

    h6_frag1 = {
        'ncas': [2, 2, 2],
        'nelecas': [2, 2, 2],
        'spin_sub': [1, 1, 1],
        'frag_atom_list': [[0, 1], [2, 3], [4, 5]]
    }

    h6_frag2 = {
        'ncas': [2, 2, 2],
        'nelecas': [2, 2, 2],
        'spin_sub': [1, 1, 1],
        'frag_atom_list': [[1,2],[3,4],[5,0]]
    }

    h8_sto3g = {
        'name': 'H8_STO3G',
        'xyz': h8_xyz,
        'basis': 'sto-3g',
    }

    h8_frag1 = {
        'ncas': [2, 2, 2, 2],
        'nelecas': [2, 2, 2, 2],
        'spin_sub': [1, 1, 1, 1],
        'frag_atom_list': [[0, 1], [2, 3], [4, 5], [6, 7]]
    }

    h8_frag2 = {
        'ncas' : [4,4],
        'nelecas' : [4,4],
        'spin_sub' : [1,1],
        'frag_atom_list' : [[0,1,2,3], [4,5,6,7]]
    }

    h10_circle_sto3g = {
        'name': 'H10_CIRCLE_STO3G',
        'xyz': h10_circle_xyz,
        'basis': 'sto-3g',
    }

    h10_circle_frag1 = {
        'ncas': [2,2,2,2,2],
        'nelecas': [2,2,2,2,2],
        'spin_sub': [1,1,1,1,1],
        'frag_atom_list': [[0,1], [2,3], [4,5], [6,7], [8,9]],
    }

    h10_circle_frag2 = {
        'ncas': [4,4,2],
        'nelecas': [4,4,2],
        'spin_sub': [1,1,1],
        'frag_atom_list': [[0,1,2,3], [4,5,6,7], [8,9]],
    }

    c4_sto3g = {
        'name': 'C4_STO3G',
        'xyz': c4xyz,
        'basis': 'sto-3g',
    }

    c4_frag1 = {
        'ncas': [2, 2],
        'nelecas': [2, 2],
        'spin_sub': [1, 1],
        'frag_atom_list': [[0, 1], [2, 3]]
    }


    c4_frag2 = {
        'ncas': [2, 2],
        'nelecas': [2, 2],
        'spin_sub': [1, 1],
        'frag_atom_list': [[1,2], [0, 3]]
    }

    c4_frag3 = {
        'ncas': [2, 2],
        'nelecas': [2, 2],
        'spin_sub': [1, 1],
        'frag_atom_list': [[0, 2], [1, 3]]
    }

    mol_confs = [c4_sto3g, h8_sto3g]
    frag_confs = [[c4_frag1, c4_frag2, c4_frag3], [h8_frag1, h8_frag2]]


    for mol_conf, frag_conf_list in zip(mol_confs, frag_confs):
        print(f"\nRunning calculation for molecule: {mol_conf['name']}")
        mol = gto.Mole(atom=mol_conf['xyz'], basis=mol_conf['basis'], verbose=0)
        
        mol.build()
        mf = scf.RHF(mol).run()
        
        las_states = build_las_states(mf, frag_conf_list)
        for i, las in enumerate(las_states):
            print(f"LASSCF state {i} energy = ", las.e_tot)

        noci_states = []
        for las in las_states:
            noci_states.append(las2noci_state(mol, las))

        h1e  = owndata(mf.get_hcore())
        h2e  = owndata(ao2mo.restore(1, mf._eri, mol.nao)
                       .reshape(mol.nao**2, mol.nao**2))
        ovlp = owndata(mf.get_ovlp())
        nmo  = mf.mo_coeff.shape[1]
        nocc = int(np.sum(mf.mo_occ > 0)) # correct for RHF

        h, s, _ = build_noci_matrices(noci_states, h1e, h2e, ovlp, nmo, nocc, mol.energy_nuc())

        # print("Hamiltonian matrix H:\n", h)
        # print("\nOverlap matrix S:\n", s)

        # ----- generalised diagonalisation -----
        evals, evecs = scipy.linalg.eigh(h, b=s)

        print("\nRecoupled NOCI energies (Eh):")
        for i, e in enumerate(evals):
            print(f"  root {i}: {e:16.10f}")

        # # (optional) display a 1‑RDM block
        # print("\n⟨Ψ₀|γ|Ψ₀⟩ (first RDM1 block):\n", rdm1[0, 0])


if __name__ == "__main__":
    main()
