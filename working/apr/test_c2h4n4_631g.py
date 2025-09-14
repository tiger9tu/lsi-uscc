#!/usr/bin/env python3
"""
Test C2H4N4 6-31G with VQE max cycles = 5
"""

import numpy as np
from scipy.linalg import eigh
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSIrq

from mrh.exploratory.citools import grad, lasci_ominus1, fockspace
from mrh.exploratory.unitary_cc import lasuccsd
from c2h4n4_struct import structure as c2h4n4_struct

def cilas2f(lasci, norb_f, nelec_f):
    """Convert LAS CI to Fock space representation"""
    ci_f = []
    for i, ci in enumerate(lasci):
        ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
    return ci_f

def get_Sij_Hij(psi_i, psi_j, h):
    """Compute overlap and Hamiltonian matrix elements"""
    ucj, hucj = psi_j.hc_x(psi_j.x, h)[1:3]
    uci, huci = psi_i.hc_x(psi_i.x, h)[1:3]
    ucj, hucj = ucj.ravel(), hucj.ravel()
    uci = uci.ravel()
    Sij = uci.conj().dot(ucj)
    Hij = uci.conj().dot(hucj)
    return Sij, Hij

def test_c2h4n4():
    """Test C2H4N4 with 6-31G basis"""
    
    print("=" * 80)
    print("C2H4N4 6-31G COMPLETE ENERGY COMPARISON") 
    print("LASSCF vs LASSIrq vs LASSIrq-VQE vs CASCI")
    print("=" * 80)
    
    # Setup C2H4N4 molecule
    mol_c2h4n4 = c2h4n4_struct(0, 0, '6-31g')
    mf = scf.RHF(mol_c2h4n4).run()
    
    # Fragment definition
    ncas_sub = (3, 3)
    nelecas_sub = ((2, 1), (1, 2))  # (nalpha, nbeta) for each fragment  
    spinsub = [1, 1]  # Singlet fragments
    frag_atom_list = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9))  # Two 5-atom fragments
    
    print(f"System: {sum(ncas_sub)} orbitals, {sum(sum(nelec) for nelec in nelecas_sub)} electrons in {len(ncas_sub)} fragments")
    print(f"Fragment atom list: {frag_atom_list}")
    print(f"RHF energy: {mf.e_tot:.10f} hartree")
    
    results = {}
    
    # 1. LASSCF
    print("1. LASSCF...")
    las = LASSCF(mf, ncas_sub, nelecas_sub, spin_sub=spinsub)
    mo_loc = las.localize_init_guess(frag_atom_list, mf.mo_coeff)
    las.kernel(mo_loc)
    results['lasscf'] = las.e_tot
    print(f"   Energy: {las.e_tot:.10f} hartree")
    
    # 2. CASCI
    print("2. CASCI...")
    ncas_total = sum(ncas_sub)
    nelec_total = sum(sum(nelec) for nelec in nelecas_sub)
    mc_casci = mcscf.CASCI(mf, ncas_total, nelec_total).set(
        fcisolver=csf_solver(mol_c2h4n4, smult=1))
    mc_casci.kernel(las.mo_coeff)
    results['casci'] = mc_casci.e_tot
    print(f"   Energy: {mc_casci.e_tot:.10f} hartree")
    
    # 3. LASSIrq
    print("3. LASSIrq[1,1]...")
    lsi = LASSIrq(las, r=1, q=1)
    e_roots, si_rq = lsi.kernel()
    results['lassirq'] = e_roots[0]
    num_states = len(e_roots)
    print(f"   Energy: {e_roots[0]:.10f} hartree ({num_states} states)")
    
    # 4. LASSIrq-VQE
    print("4. LASSIrq-VQE (max_cycle=5)...")
    try:
        # Get electron configurations
        nelec_fr = lsi.get_nelec_frs()
        ncore, ncas = las.ncore, las.ncas
        nelecas = sum(las.nelecas_sub)
        mo_coeff = las.mo_coeff
        
        # VQE optimization for each CT state
        nstates = len(lsi.ci[0])
        print(f"   Number of CT states: {nstates}")
        psis = []
        vqe_individual_energies = []  # Store individual VQE energies
        
        for i in range(nstates):
            ci = [[lsi.ci[fragj][i]] for fragj in range(len(lsi.ci))]
            nelec_sub = [nelec_fr[fragj][i] for fragj in range(len(nelec_fr))]
            
            # Temporary LASSCF for gradient
            tmplas = LASSCF(mf, ncas_sub, nelec_sub)
            tmplas.mo_coeff = mo_coeff
            tmplas.ci = ci
            all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(tmplas, 0.00001)
            
            # VQE solver setup
            nelecas_sub = [sum(nelec) for nelec in nelec_sub]
            mc_uscc = mcscf.CASCI(mf, ncas, nelecas)
            mc_uscc.mo_coeff = mo_coeff
            mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol_c2h4n4, a_idxs_selected, i_idxs_selected)
            mc_uscc.fcisolver.norb_f = las.ncas_sub
            lasci_ominus1.GLOBAL_MAX_CYCLE = 5  # Set to 5 as requested
            mc_uscc.kernel(ci0=cilas2f(ci, las.ncas_sub, nelec_sub))
            psis.append(mc_uscc.fcisolver.psi)
            vqe_individual_energies.append(mc_uscc.e_tot)  # Store individual VQE energy
        
        # Matrix elements between VQE states
        h1eff, e_core = mc_uscc.get_h1eff(mc_uscc.mo_coeff)
        h2eff = mc_uscc.get_h2eff()
        h = [e_core, h1eff, h2eff]
        
        nc = len(psis)
        S = np.zeros((nc, nc), dtype=np.complex128)
        H = np.zeros((nc, nc), dtype=np.complex128)
        
        for i in range(nc):
            for j in range(nc):
                S[i, j], H[i, j] = get_Sij_Hij(psis[i], psis[j], h)
        
        print("   Overlap matrix S:")
        for i, row in enumerate(S):
            print(f"     Row {i}: " + " ".join([f"{val.real:8.6f}" for val in row]))
        print("   Hamiltonian matrix H:")
        for i, row in enumerate(H):
            print(f"     Row {i}: " + " ".join([f"{val.real:10.8f}" for val in row]))
        
        # Find lowest individual VQE energy before state interaction
        lowest_individual_vqe = min(vqe_individual_energies)
        print(f"   Individual VQE energies: {[f'{e:.10f}' for e in vqe_individual_energies]}")
        print(f"   Lowest individual VQE energy: {lowest_individual_vqe:.10f} hartree")
        
        # Solve eigenvalue problem
        eigvals, eigvecs = eigh(H, S)
        results['lassirq_vqe'] = eigvals[0]
        results['lowest_individual_vqe'] = lowest_individual_vqe
        print(f"   Final LASSIrq-VQE energy (after state interaction): {eigvals[0]:.10f} hartree")
        
        vqe_si_improvement = eigvals[0] - lowest_individual_vqe
        print(f"   State interaction improvement: {vqe_si_improvement:+.10f} hartree")
        
    except Exception as e:
        print(f"   LASSIrq-VQE failed: {e}")
        import traceback
        traceback.print_exc()
        results['lassirq_vqe'] = None
        results['lowest_individual_vqe'] = None
    
    # Summary
    print(f"\n{'='*80}")
    print("ENERGY SUMMARY FOR C2H4N4_631G")
    print(f"{'='*80}")
    
    print(f"{'Method':<15} {'Energy (hartree)':<20} {'Rel to LASSCF (mH)':<20}")
    print("-" * 60)
    
    lasscf_energy = results['lasscf']
    methods = [('casci', 'CASCI'), ('lassirq_vqe', 'LASSIrq-VQE'), 
               ('lassirq', 'LASSIrq'), ('lasscf', 'LASSCF')]
    
    for method, label in methods:
        if results.get(method) is not None:
            energy = results[method]
            rel_energy = (energy - lasscf_energy) * 1000  # Convert to milli-hartree
            print(f"{label:<15} {energy:<20.10f} {rel_energy:+18.6f}")
    
    if results.get('lowest_individual_vqe'):
        energy = results['lowest_individual_vqe']
        rel_energy = (energy - lasscf_energy) * 1000
        print(f"{'VQE (indiv)':<15} {energy:<20.10f} {rel_energy:+18.6f}")
    
    # Energy ordering
    energies = []
    labels = []
    for method, label in methods:
        if results.get(method) is not None:
            energies.append(results[method])
            labels.append(label)
    
    if len(energies) >= 3:
        sorted_pairs = sorted(zip(energies, labels))
        print(f"\nEnergy ordering (lowest to highest):")
        for i, (energy, label) in enumerate(sorted_pairs):
            print(f"  {i+1}. {label:<15}: {energy:.10f} hartree")
        
        # Check correct ordering
        if (results.get('casci') is not None and 
            results.get('lassirq_vqe') is not None and 
            results.get('lassirq') is not None):
            correct = (results['casci'] <= results['lassirq_vqe'] <= 
                      results['lassirq'] <= results['lasscf'])
            print(f"\nCorrect ordering (CASCI ≤ LASSIrq-VQE ≤ LASSIrq ≤ LASSCF): {'✓' if correct else '✗'}")
    
    return results

if __name__ == "__main__":
    results = test_c2h4n4()