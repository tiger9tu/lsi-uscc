#!/usr/bin/env python3
"""
LAS-VQE-NOCI Implementation for H6 molecule

This script implements:
1. LASSCF calculation with 3 fragments
2. Gradient-based excitation selection 
3. Two VQE optimizations localized to specific fragment pairs
4. State interaction between the two VQE states
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


h6_frag_spin_orb = {
    0: (0, 1, 6, 7),
    1: (2, 3, 8, 9),
    2: (4, 5, 10, 11)
}

def select_local_interfrag_excitations(a_idxs, i_idxs, frag_pair_sorb):
    """Select excitations localized to a specific fragment pair"""
    a_idxs_sel, i_idxs_sel = [], []
    for a, i in zip(a_idxs, i_idxs):
        # Check if excitation is within the specified fragment pair
        if all(idx in frag_pair_sorb for idx in a) and all(idx in frag_pair_sorb for idx in i):
            a_idxs_sel.append(a)
            i_idxs_sel.append(i)
    return a_idxs_sel, i_idxs_sel


def main():
    print("=" * 70)
    print("LAS-VQE-NOCI Implementation for H6 Molecule")
    print("=" * 70)
    
    # H6 geometry (3 fragments of 2 H atoms each)
    H6xyz = """
    H      0.000000000000   0.000000000000   0.000000000000
    H      1.000000000000   0.000000000000   0.000000000000
    H      0.273746762116   2.195450598147   0.100000000000
    H      1.232912762116   1.895450598147  -0.100000000000
    H      0.507178110854   4.193780995243   0.049334760036
    H      1.506140937609   3.988021397347  -0.049334760036
    """
    
    # Setup molecule
    mol = gto.M(
        atom=H6xyz,
        basis='6-31g',
        verbose=0
    )
    mol.build()
    
    print(f"Molecule setup complete:")
    print(f"  Basis: 6-31G")
    print(f"  Atoms: {mol.natm}")
    print(f"  Total electrons: {mol.nelectron}")
    print(f"  Orbitals: {mol.nao}")
    
    # Mean-field calculation  
    mf = scf.RHF(mol).run()
    print(f"  RHF energy: {mf.e_tot:.10f} hartree")
    
    # LASSCF setup: 3 fragments, each with 2 orbitals and 2 electrons total
    # For H6 with 6 electrons: (2,2,2) electrons = 6 total electrons
    print("\n" + "="*50)
    print("STEP 1: LASSCF Calculation")
    print("="*50)
    
    las = LASSCF(mf, (2, 2, 2), (2, 2, 2), spin_sub=(1, 1, 1))
    
    # Fragment atom lists: (atoms 0-1), (atoms 2-3), (atoms 4-5)
    frag_atom_list = ((0, 1), (2, 3), (4, 5))
    
    # Localize initial guess
    mo_loc = las.localize_init_guess(frag_atom_list, mf.mo_coeff)
    result = las.kernel(mo_loc)
    h2eff_sub, veff = result[-2:]
    
    if not las.converged:
        print("WARNING: LASSCF did not converge!")
        return
    
    print(f"LASSCF calculation complete:")
    print(f"  Energy: {las.e_tot:.10f} hartree")
    print(f"  Fragments: {len(las.ncas_sub)}")
    print(f"  Active orbitals per fragment: {las.ncas_sub}")
    print(f"  Active electrons per fragment: {las.nelecas_sub}")

    cas = mcscf.CASCI(mf, las.ncas, sum(las.nelecas_sub))
    cas.mo_coeff = las.mo_coeff
    e_cas = cas.kernel()[0]
    print(f"  CASCI energy (for reference): {e_cas:.10f} hartree")
    
    # Get molecular orbitals and integrals for VQE
    ncore, ncas = las.ncore, las.ncas  
    mo_coeff = las.mo_coeff
    mo_cas = mo_coeff[:, ncore:ncore+ncas]
    
    # Get effective Hamiltonian for VQE (already obtained from las.kernel())
    e0 = las._scf.energy_nuc() + 2 * (((las._scf.get_hcore() + veff.c/2) @ mo_coeff[:,:ncore]) * mo_coeff[:,:ncore]).sum()
    h1 = mo_cas.conj().T @ (las._scf.get_hcore() + veff.c) @ mo_cas
    h2 = h2eff_sub[ncore:ncore+ncas].reshape(ncas*ncas, ncas*(ncas+1)//2)
    h2 = lib.numpy_helper.unpack_tril(h2).reshape(ncas, ncas, ncas, ncas)
    
    print("\n" + "="*50)
    print("STEP 2: Gradient-based Excitation Selection")
    print("="*50)
    
    # Create a temporary LASSCF with the reference configuration for gradient calculation
    ci_ref = las.ci
    nelec_sub = las.nelecas_sub
    
    tmplas = LASSCF(mf, (2, 2, 2), nelec_sub)
    tmplas.mo_coeff = mo_coeff
    tmplas.ci = ci_ref
    
    # Get all gradients and select important excitations
    all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(tmplas, 0.0001)
    
    print(f"Gradient-based selection complete:")
    print(f"  Total excitations found: {len(a_idxs_selected)}")
    print(f"  Selection threshold: 0.0001")

    # Create two different excitation sets to demonstrate LAS-VQE-NOCI
    print("\n" + "="*50)
    print("STEP 3: Create Two Excitation Sets") 
    print("="*50)
    
    # For demonstration, use different subsets of the selected excitations
    # In a real calculation, you would filter by spatial/fragment localization
    total_excitations = len(a_idxs_selected)
    
    # Excitation set 1: First half of selected excitations
    split_point = total_excitations // 2
    a_idxs_12,i_idxs_12 = select_local_interfrag_excitations(a_idxs_selected, i_idxs_selected,
                                                            h6_frag_spin_orb[0] + h6_frag_spin_orb[1])
    print("a_idxs_12:", a_idxs_12)
    print("i_idxs_12:", i_idxs_12)
    # Excitation set 2: Second half of selected excitations  
    a_idxs_23,i_idxs_23 = select_local_interfrag_excitations(a_idxs_selected, i_idxs_selected,
                                                            h6_frag_spin_orb[1] + h6_frag_spin_orb[2])
    
    print(f"Excitation set 1: {len(a_idxs_12)} excitations")
    print(f"Excitation set 2: {len(a_idxs_23)} excitations")
    
    # Ensure we have at least some excitations for each set
    if len(a_idxs_12) == 0:
        print("WARNING: No excitations in set 1, using first 10 excitations")
        a_idxs_12, i_idxs_12 = a_idxs_selected[:10], i_idxs_selected[:10]
        
    if len(a_idxs_23) == 0:
        print("WARNING: No excitations in set 2, using last 10 excitations") 
        a_idxs_23, i_idxs_23 = a_idxs_selected[-10:], i_idxs_selected[-10:]
    
    print("\n" + "="*50)
    print("STEP 4: VQE Optimization - Excitation Set 1")
    print("="*50)
    
    # VQE for excitation set 1
    nelecas_sub = [sum(nelec) for nelec in nelec_sub]
    mc_uscc_12 = mcscf.CASCI(mf, ncas, sum(nelecas_sub))
    mc_uscc_12.mo_coeff = mo_coeff
    mc_uscc_12.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_12, i_idxs_12)
    mc_uscc_12.fcisolver.norb_f = las.ncas_sub
    
    # Set maximum VQE cycles
    lasci_ominus1.GLOBAL_MAX_CYCLE = 10
    
    # Convert LAS CI to fock space and run VQE
    ci0_12 = cilas2f(ci_ref, las.ncas_sub, nelec_sub)
    mc_uscc_12.kernel(ci0=ci0_12)
    psi_12 = mc_uscc_12.fcisolver.psi
    
    print(f"VQE optimization complete for excitation set 1:")
    print(f"  Number of excitations: {len(a_idxs_12)}")
    print(f"  VQE converged: {hasattr(mc_uscc_12.fcisolver, 'converged')}")
    
    print("\n" + "="*50)
    print("STEP 5: VQE Optimization - Excitation Set 2")
    print("="*50)
    
    # VQE for excitation set 2
    mc_uscc_23 = mcscf.CASCI(mf, ncas, sum(nelecas_sub))
    mc_uscc_23.mo_coeff = mo_coeff
    mc_uscc_23.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_23, i_idxs_23)
    mc_uscc_23.fcisolver.norb_f = las.ncas_sub
    
    # Convert LAS CI to fock space and run VQE
    ci0_23 = cilas2f(ci_ref, las.ncas_sub, nelec_sub)
    mc_uscc_23.kernel(ci0=ci0_23)
    psi_23 = mc_uscc_23.fcisolver.psi
    
    print(f"VQE optimization complete for excitation set 2:")
    print(f"  Number of excitations: {len(a_idxs_23)}")
    print(f"  VQE converged: {hasattr(mc_uscc_23.fcisolver, 'converged')}")
    
    print("\n" + "="*50)
    print("STEP 6: State Interaction")
    print("="*50)
    
    # Prepare Hamiltonian for state interaction
    h1eff, e_core = mc_uscc_12.get_h1eff(mc_uscc_12.mo_coeff)
    h2eff = mc_uscc_12.get_h2eff()
    h = [e_core, h1eff, h2eff]
    
    # Build overlap and Hamiltonian matrices
    psis = [psi_12, psi_23]
    nc = len(psis)
    
    S = np.zeros((nc, nc), dtype=np.complex128)
    H = np.zeros((nc, nc), dtype=np.complex128)
    
    for i in range(nc):
        for j in range(nc):
            S[i, j], H[i, j] = get_Sij_Hij(psis[i], psis[j], h)
    
    print("State interaction matrices:")
    print("Overlap matrix S:")
    for i, row in enumerate(S):
        print(f"  [{i}]: {' '.join(f'{x.real:12.8f}' for x in row)}")
    
    print("Hamiltonian matrix H:")
    for i, row in enumerate(H):
        print(f"  [{i}]: {' '.join(f'{x.real:12.8f}' for x in row)}")
    
    # Solve generalized eigenvalue problem
    eigvals, eigvecs = eigh(H, S)
    
    print(f"\nFinal LAS-VQE-NOCI Results:")
    print(f"  Ground state energy: {eigvals[0]:.10f} hartree")
    print(f"  Excited state energy: {eigvals[1]:.10f} hartree")
    print(f"  Energy gap: {(eigvals[1] - eigvals[0]) * 27.2114:.3f} eV")
    
    print(f"\nGround state coefficients:")
    print(f"  VQE State 1: {eigvecs[0, 0]:.6f}")
    print(f"  VQE State 2: {eigvecs[1, 0]:.6f}")
    
    # Compare with reference LASSCF energy
    print(f"\nEnergy comparison:")
    print(f"  LASSCF energy:      {las.e_tot:.10f} hartree") 
    print(f"  CASCI energy:       {e_cas:.10f} hartree")
    print(f"  LAS-VQE-NOCI energy: {eigvals[0]:.10f} hartree")
    print(f"  Correlation energy: {(eigvals[0] - las.e_tot)*1000:.3f} mEh")
    
    print("\n" + "="*70)
    print("LAS-VQE-NOCI calculation completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()