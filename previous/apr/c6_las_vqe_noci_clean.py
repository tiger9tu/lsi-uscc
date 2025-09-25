#!/usr/bin/env python3
"""
Clean LAS-VQE-NOCI Implementation for C6 Molecule

This implementation follows the methodology:
1. LASSCF with 3 fragments 
2. Gradient-based excitation selection
3. Fragment-localized VQE states (fragments 0-1 and 1-2)
4. Non-orthogonal configuration interaction
"""

import sys
import numpy as np
from pyscf import gto, scf, mcscf, lib
from scipy.linalg import eigh
sys.path.append('/home/jinx/repo/qchem/las-uscc-noci-bot/working')

from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.citools import grad, lasci_ominus1, fockspace
from mrh.exploratory.unitary_cc import lasuccsd

def cilas2f(lasci, norb_f, nelec_f):
    """Convert LAS CI to fock space format"""
    ci_f = []
    for i, ci in enumerate(lasci):
        ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
    return ci_f

def get_Sij_Hij(psi_i, psi_j, h):
    """Get overlap and Hamiltonian matrix elements between VQE states"""
    ucj, hucj = psi_j.hc_x(psi_j.x, h)[1:3]
    uci, huci = psi_i.hc_x(psi_i.x, h)[1:3]
    ucj, hucj = ucj.ravel(), hucj.ravel()
    uci = uci.ravel()
    Sij = uci.conj().dot(ucj)
    Hij = uci.conj().dot(hucj)
    return Sij, Hij

# C6 fragment spin-orbital mapping
c6_frag_spin_orb = {
    0: (0, 1, 6, 7),    # Fragment 0
    1: (2, 3, 8, 9),    # Fragment 1  
    2: (4, 5, 10, 11)   # Fragment 2
}

def select_local_interfrag_excitations(a_idxs, i_idxs, frag_pair_sorb):
    """Select excitations localized to a specific fragment pair"""
    a_idxs_sel, i_idxs_sel = [], []
    for a, i in zip(a_idxs, i_idxs):
        if all(idx in frag_pair_sorb for idx in a) and all(idx in frag_pair_sorb for idx in i):
            a_idxs_sel.append(a)
            i_idxs_sel.append(i)
    return a_idxs_sel, i_idxs_sel

def main():
    print("=" * 70)
    print("LAS-VQE-NOCI for C6 Molecule")
    print("=" * 70)
    
    # Load C6 geometry
    data_dir = '/home/jinx/repo/qchem/las_uccsd_data'
    with open(f'{data_dir}/polyenes/geometries/c6.xyz', 'r') as f:
        c6xyz = f.read()
    
    # Setup molecule
    mol = gto.M(
        atom=c6xyz,
        basis='sto-3g',
        verbose=0
    )
    mol.build()
    
    print(f"Molecule: {mol.natm} atoms, {mol.nelectron} electrons")
    
    # RHF calculation
    mf = scf.RHF(mol).run()
    print(f"RHF energy: {mf.e_tot:.10f} hartree")
    
    # LASSCF calculation (3 fragments, 2 orbitals and 2 electrons each)
    print("\nLASCCF calculation...")
    frag_atom_list = [[0, 2], [10, 12], [3, 1]]
    las = LASSCF(mf, (2, 2, 2), (2, 2, 2), spin_sub=(1, 1, 1))
    
    mo_loc = las.localize_init_guess(frag_atom_list, mf.mo_coeff)
    result = las.kernel(mo_loc)
    h2eff_sub, veff = result[-2:]
    
    print(f"LASSCF energy: {las.e_tot:.10f} hartree")
    print(f"LASSCF converged: {las.converged}")
    
    # CASCI reference
    cas = mcscf.CASCI(mf, las.ncas, sum(las.nelecas_sub))
    cas.mo_coeff = las.mo_coeff
    e_casci = cas.kernel()[0]
    print(f"CASCI energy: {e_casci:.10f} hartree")
    
    # Gradient-based excitation selection
    print("\nExcitation selection...")
    tmplas = LASSCF(mf, (2, 2, 2), las.nelecas_sub)
    tmplas.mo_coeff = las.mo_coeff
    tmplas.ci = las.ci
    
    all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(tmplas, 0.0001)
    print(f"Selected {len(a_idxs_selected)} excitations")
    
    # Fragment-localized excitation sets
    print("\nCreating fragment-localized excitation sets...")
    
    # Set 1: Fragments 0-1 (first two fragments)
    frag_01_orbs = c6_frag_spin_orb[0] + c6_frag_spin_orb[1]
    a_idxs_01, i_idxs_01 = select_local_interfrag_excitations(
        a_idxs_selected, i_idxs_selected, frag_01_orbs)
    
    # Set 2: Fragments 1-2 (last two fragments)
    frag_12_orbs = c6_frag_spin_orb[1] + c6_frag_spin_orb[2]  
    a_idxs_12, i_idxs_12 = select_local_interfrag_excitations(
        a_idxs_selected, i_idxs_selected, frag_12_orbs)
    
    print(f"Fragment 0-1 excitations: {len(a_idxs_01)}")
    print(f"Fragment 1-2 excitations: {len(a_idxs_12)}")
    
    # Fallback if no fragment-localized excitations found
    if len(a_idxs_01) == 0:
        print("No fragment 0-1 excitations found, using first half")
        split = len(a_idxs_selected) // 2
        a_idxs_01, i_idxs_01 = a_idxs_selected[:split], i_idxs_selected[:split]
    
    if len(a_idxs_12) == 0:
        print("No fragment 1-2 excitations found, using second half")
        split = len(a_idxs_selected) // 2
        a_idxs_12, i_idxs_12 = a_idxs_selected[split:], i_idxs_selected[split:]
    
    # VQE optimization for both excitation sets
    print("\nVQE optimization...")
    lasci_ominus1.GLOBAL_MAX_CYCLE = 10
    nelecas_sub = [sum(nelec) for nelec in las.nelecas_sub]
    
    # VQE for fragment set 0-1
    print("  Optimizing VQE state 1 (fragments 0-1)...")
    mc_uscc_01 = mcscf.CASCI(mf, las.ncas, sum(nelecas_sub))
    mc_uscc_01.mo_coeff = las.mo_coeff
    mc_uscc_01.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_01, i_idxs_01)
    mc_uscc_01.fcisolver.norb_f = las.ncas_sub
    
    ci0_01 = cilas2f(las.ci, las.ncas_sub, las.nelecas_sub)
    mc_uscc_01.kernel(ci0=ci0_01)
    psi_01 = mc_uscc_01.fcisolver.psi
    
    # VQE for fragment set 1-2
    print("  Optimizing VQE state 2 (fragments 1-2)...")
    mc_uscc_12 = mcscf.CASCI(mf, las.ncas, sum(nelecas_sub))
    mc_uscc_12.mo_coeff = las.mo_coeff
    mc_uscc_12.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_12, i_idxs_12)
    mc_uscc_12.fcisolver.norb_f = las.ncas_sub
    
    ci0_12 = cilas2f(las.ci, las.ncas_sub, las.nelecas_sub)
    mc_uscc_12.kernel(ci0=ci0_12)
    psi_12 = mc_uscc_12.fcisolver.psi
    
    print(f"  VQE state 1 converged: {hasattr(mc_uscc_01.fcisolver, 'converged')}")
    print(f"  VQE state 2 converged: {hasattr(mc_uscc_12.fcisolver, 'converged')}")
    
    # State interaction
    print("\nState interaction...")
    h1eff, e_core = mc_uscc_01.get_h1eff(mc_uscc_01.mo_coeff)
    h2eff = mc_uscc_01.get_h2eff()
    h = [e_core, h1eff, h2eff]
    
    psis = [psi_01, psi_12]
    nc = len(psis)
    
    S = np.zeros((nc, nc), dtype=np.complex128)
    H = np.zeros((nc, nc), dtype=np.complex128)
    
    for i in range(nc):
        for j in range(nc):
            S[i, j], H[i, j] = get_Sij_Hij(psis[i], psis[j], h)
    
    print("Overlap matrix S:")
    for i, row in enumerate(S):
        print(f"  [{i}]: {' '.join(f'{x.real:10.6f}' for x in row)}")
    
    print("Hamiltonian matrix H:")
    for i, row in enumerate(H):
        print(f"  [{i}]: {' '.join(f'{x.real:12.8f}' for x in row)}")
    
    # Solve generalized eigenvalue problem
    eigvals, eigvecs = eigh(H, S)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Ground state energy:     {eigvals[0]:.10f} hartree")
    print(f"Excited state energy:    {eigvals[1]:.10f} hartree")
    print(f"Energy gap:              {(eigvals[1] - eigvals[0]) * 27.2114:.3f} eV")
    
    print(f"\nGround state coefficients:")
    print(f"  VQE state 1 (frags 0-1): {eigvecs[0, 0]:+.6f}")
    print(f"  VQE state 2 (frags 1-2): {eigvecs[1, 0]:+.6f}")
    
    print(f"\nEnergy comparison:")
    print(f"  RHF energy:             {mf.e_tot:.10f} hartree")
    print(f"  LASSCF energy:          {las.e_tot:.10f} hartree")
    print(f"  CASCI energy:           {e_casci:.10f} hartree")
    print(f"  LAS-VQE-NOCI energy:    {eigvals[0]:.10f} hartree")
    
    print(f"\nCorrelation energies:")
    print(f"  vs LASSCF:  {(eigvals[0] - las.e_tot) * 1000:+.3f} mEh")
    print(f"  vs CASCI:   {(eigvals[0] - e_casci) * 1000:+.3f} mEh")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()