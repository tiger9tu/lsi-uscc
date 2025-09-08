#!/usr/bin/env python3
"""
Complete Energy Comparison: LASSCF vs LASSIrq vs LASSIrq-VQE vs CASCI

This script implements the corrected LASSIrq-VQE based on mylassi.py and runs all four methods.
"""

import numpy as np
import time
from scipy.linalg import eigh
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSIrq

from mrh.exploratory.citools import grad, lasci_ominus1, fockspace
from mrh.exploratory.unitary_cc import lasuccsd

from lassi_instances_lassirq import molecular_configs, setup_molecule

def cilas2f(lasci, norb_f, nelec_f):
    """Convert LAS CI to Fock space representation (from mylassi.py)"""
    ci_f = []
    for i, ci in enumerate(lasci):
        ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
    return ci_f

def get_Sij_Hij(psi_i, psi_j, h):
    """Compute overlap and Hamiltonian matrix elements (from mylassi.py)"""
    ucj, hucj = psi_j.hc_x(psi_j.x, h)[1:3]
    uci, huci = psi_i.hc_x(psi_i.x, h)[1:3]
    ucj, hucj = ucj.ravel(), hucj.ravel()
    uci = uci.ravel()
    Sij = uci.conj().dot(ucj)
    Hij = uci.conj().dot(hucj)
    return Sij, Hij

def calculate_lassirq_vqe_energy(las, lsi, mol, mf, max_cycle=3):
    """
    Calculate LASSIrq-VQE energy using the correct implementation from mylassi.py
    """
    try:
        # Get necessary parameters
        ncore, ncas = las.ncore, las.ncas
        nelecas = sum(las.nelecas_sub)
        mo_coeff = las.mo_coeff
        nelec_fr = lsi.get_nelec_frs()
        
        # VQE optimization for each CT state
        nstates = len(lsi.ci[0])
        psis = []
        
        print(f"    Building VQE wavefunctions for {nstates} states...")
        
        for i in range(nstates):
            # Extract CI coefficients and electron configuration for state i
            ci = [[lsi.ci[fragj][i]] for fragj in range(len(lsi.ci))]
            nelec_sub = [nelec_fr[fragj][i] for fragj in range(len(nelec_fr))]
            
            # Create temporary LASSCF object for gradient calculation
            tmplas = LASSCF(mf, las.ncas_sub, nelec_sub)
            tmplas.mo_coeff = mo_coeff
            tmplas.ci = ci
            
            # Get gradient information for VQE optimization
            try:
                all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(tmplas, 0.00001)
            except Exception as e:
                print(f"      Warning: Gradient calculation failed for state {i}: {e}")
                a_idxs_selected, i_idxs_selected = [], []
            
            # Setup VQE solver 
            nelecas_sub = [sum(nelec) for nelec in nelec_sub]
            mc_uscc = mcscf.CASCI(mf, ncas, nelecas)
            mc_uscc.mo_coeff = mo_coeff
            mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
            mc_uscc.fcisolver.norb_f = las.ncas_sub
            
            # Set VQE parameters
            lasci_ominus1.GLOBAL_MAX_CYCLE = max_cycle
            
            # Run VQE optimization
            ci0 = cilas2f(ci, las.ncas_sub, nelec_sub)
            mc_uscc.kernel(ci0=ci0)
            psis.append(mc_uscc.fcisolver.psi)
        
        # Get effective Hamiltonian
        h1eff, e_core = mc_uscc.get_h1eff(mc_uscc.mo_coeff)
        h2eff = mc_uscc.get_h2eff()
        h = [e_core, h1eff, h2eff]
        
        # Compute matrix elements between VQE states
        nc = len(psis)
        S = np.zeros((nc, nc), dtype=np.complex128)
        H = np.zeros((nc, nc), dtype=np.complex128)
        
        for i in range(nc):
            for j in range(nc):
                S[i, j], H[i, j] = get_Sij_Hij(psis[i], psis[j], h)
        
        # Solve generalized eigenvalue problem
        eigvals, eigvecs = eigh(H, S)
        
        return eigvals[0]  # Ground state energy
        
    except Exception as e:
        print(f"    LASSIrq-VQE failed: {e}")
        return None

def calculate_all_energies(mol_config):
    """Calculate all four energies for a single molecule"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {mol_config['name']}")
    print(f"{'='*60}")
    
    results = {
        'name': mol_config['name'],
        'ncas': mol_config['ncas'],
        'nelecas': mol_config['nelecas'],
        'rhf_energy': None,
        'lasscf_energy': None,
        'casci_energy': None,
        'lassirq_energy': None,
        'lassirq_vqe_energy': None,
        'num_lassirq_states': None,
        'notes': []
    }
    
    try:
        # Setup molecule
        mol, mf = setup_molecule(mol_config)
        ncas_total = sum(mol_config['ncas'])
        nelec_total = sum(mol_config['nelecas'])
        results['rhf_energy'] = mf.e_tot
        
        print(f"System: {ncas_total} orbitals, {nelec_total} electrons in {len(mol_config['ncas'])} fragments")
        print(f"RHF energy: {mf.e_tot:.10f} hartree")
        
        # 1. LASSCF calculation
        print(f"1. Running LASSCF...")
        las = LASSCF(mf, mol_config['ncas'], mol_config['nelecas'], 
                     spin_sub=mol_config['spinsub'])
        mo_loc = las.localize_init_guess(mol_config['frag_atom_list'], mf.mo_coeff)
        las.kernel(mo_loc)
        
        if not las.converged:
            print(f"   LASSCF failed to converge")
            results['notes'].append("LASSCF_FAILED")
            return results
            
        results['lasscf_energy'] = las.e_tot
        print(f"   LASSCF energy: {las.e_tot:.10f} hartree")
        
        # 2. CASCI calculation
        print(f"2. Running CASCI reference...")
        try:
            mc_casci = mcscf.CASCI(mf, ncas_total, nelec_total).set(
                fcisolver=csf_solver(mol, smult=1))
            mc_casci.kernel(las.mo_coeff)
            if mc_casci.converged:
                results['casci_energy'] = mc_casci.e_tot
                print(f"   CASCI energy: {mc_casci.e_tot:.10f} hartree")
            else:
                print(f"   CASCI failed to converge")
                results['notes'].append("CASCI_FAILED")
        except Exception as e:
            print(f"   CASCI failed: {e}")
            results['notes'].append("CASCI_FAILED")
        
        # 3. LASSIrq calculation
        print(f"3. Running LASSIrq[1,1]...")
        try:
            lsi = LASSIrq(las, r=1, q=1)
            e_roots, si_rq = lsi.kernel()
            results['lassirq_energy'] = e_roots[0]
            results['num_lassirq_states'] = len(e_roots)
            print(f"   LASSIrq energy: {e_roots[0]:.10f} hartree")
            print(f"   Number of states: {len(e_roots)}")
            
            # 4. LASSIrq-VQE calculation
            print(f"4. Running LASSIrq-VQE (max_cycle=3)...")
            lassirq_vqe_energy = calculate_lassirq_vqe_energy(las, lsi, mol, mf, max_cycle=3)
            
            if lassirq_vqe_energy is not None:
                results['lassirq_vqe_energy'] = lassirq_vqe_energy
                print(f"   LASSIrq-VQE energy: {lassirq_vqe_energy:.10f} hartree")
            else:
                results['notes'].append("LASSIRQ_VQE_FAILED")
                
        except Exception as e:
            print(f"   LASSIrq failed: {e}")
            results['notes'].append("LASSIRQ_FAILED")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"ENERGY SUMMARY FOR {mol_config['name']}")
        print(f"{'='*60}")
        
        energies = []
        labels = []
        
        if results['casci_energy'] is not None:
            energies.append(results['casci_energy'])
            labels.append("CASCI")
            print(f"CASCI (lower bound):      {results['casci_energy']:15.10f} hartree")
        
        if results['lasscf_energy'] is not None:
            energies.append(results['lasscf_energy'])
            labels.append("LASSCF")
            print(f"LASSCF:                   {results['lasscf_energy']:15.10f} hartree")
        
        if results['lassirq_energy'] is not None:
            energies.append(results['lassirq_energy'])
            labels.append("LASSIrq")
            print(f"LASSIrq[1,1]:             {results['lassirq_energy']:15.10f} hartree")
            if results['lasscf_energy'] is not None:
                diff = results['lassirq_energy'] - results['lasscf_energy']
                print(f"  (vs LASSCF:             {diff:+15.10f} hartree)")
        
        if results['lassirq_vqe_energy'] is not None:
            energies.append(results['lassirq_vqe_energy'])
            labels.append("LASSIrq-VQE")
            print(f"LASSIrq-VQE:              {results['lassirq_vqe_energy']:15.10f} hartree")
            if results['lassirq_energy'] is not None:
                diff = results['lassirq_vqe_energy'] - results['lassirq_energy']
                print(f"  (vs LASSIrq:            {diff:+15.10f} hartree)")
        
        # Energy ordering
        if len(energies) >= 2:
            sorted_indices = np.argsort(energies)
            print(f"\nEnergy ordering (lowest to highest):")
            for i, idx in enumerate(sorted_indices):
                print(f"  {i+1}. {labels[idx]:15s} {energies[idx]:15.10f} hartree")
        
        return results
        
    except Exception as e:
        print(f"ERROR: {e}")
        results['notes'].append("GENERAL_FAILURE")
        return results

def main():
    """Run complete comparison on all molecular instances"""
    print("=" * 80)
    print("COMPLETE ENERGY COMPARISON: LASSCF vs LASSIrq vs LASSIrq-VQE vs CASCI")
    print("=" * 80)
    
    all_results = []
    total_calcs = len(molecular_configs)
    
    # Process all molecules (limit to first few for testing)
    test_molecules = molecular_configs[:6]  # Test first 6 molecules
    
    for i, mol_config in enumerate(test_molecules):
        print(f"\nProgress: {i+1}/{len(test_molecules)} molecules")
        
        results = calculate_all_energies(mol_config)
        all_results.append(results)
    
    # Summary table
    print(f"\n{'='*80}")
    print("FINAL SUMMARY TABLE")
    print(f"{'='*80}")
    
    header = f"{'Molecule':<15} {'LASSCF':<12} {'LASSIrq':<12} {'LASSIrq-VQE':<12} {'CASCI':<12} {'States':<7} {'Status':<15}"
    print(header)
    print("-" * len(header))
    
    for result in all_results:
        row = f"{result['name']:<15} "
        row += f"{result['lasscf_energy']:.6f}" if result['lasscf_energy'] else "FAILED      "
        row += " "
        row += f"{result['lassirq_energy']:.6f}" if result['lassirq_energy'] else "FAILED      "
        row += " "
        row += f"{result['lassirq_vqe_energy']:.6f}" if result['lassirq_vqe_energy'] else "FAILED      "
        row += " "
        row += f"{result['casci_energy']:.6f}" if result['casci_energy'] else "FAILED      "
        row += " "
        row += f"{result['num_lassirq_states']:<7}" if result['num_lassirq_states'] else "N/A    "
        row += " "
        row += "; ".join(result['notes']) if result['notes'] else "SUCCESS"
        print(row)
    
    # Energy differences analysis for successful calculations
    successful_results = [r for r in all_results if all([
        r['lasscf_energy'] is not None,
        r['lassirq_energy'] is not None,
        r['casci_energy'] is not None,
        r['lassirq_vqe_energy'] is not None
    ])]
    
    if successful_results:
        print(f"\n{'='*80}")
        print("ENERGY DIFFERENCES ANALYSIS")
        print(f"{'='*80}")
        
        print(f"Analysis for {len(successful_results)} molecules with all methods successful:")
        print(f"{'Molecule':<15} {'LASSIrq-LASSCF':<15} {'VQE-LASSIrq':<15} {'VQE-LASSCF':<15} {'CASCI-VQE':<15}")
        print("-" * 80)
        
        for r in successful_results:
            lassirq_lasscf = r['lassirq_energy'] - r['lasscf_energy']
            vqe_lassirq = r['lassirq_vqe_energy'] - r['lassirq_energy']
            vqe_lasscf = r['lassirq_vqe_energy'] - r['lasscf_energy']
            casci_vqe = r['casci_energy'] - r['lassirq_vqe_energy']
            
            print(f"{r['name']:<15} {lassirq_lasscf:+14.8f} {vqe_lassirq:+14.8f} {vqe_lasscf:+14.8f} {casci_vqe:+14.8f}")
        
        print(f"\nExpected energy ordering: CASCI < LASSIrq-VQE < LASSIrq < LASSCF")
        
        # Check energy ordering
        correct_ordering = 0
        for r in successful_results:
            if (r['casci_energy'] <= r['lassirq_vqe_energy'] <= 
                r['lassirq_energy'] <= r['lasscf_energy']):
                correct_ordering += 1
        
        print(f"Molecules with correct energy ordering: {correct_ordering}/{len(successful_results)}")
    
    print(f"\n{'='*80}")
    print(f"CALCULATION SUMMARY")
    print(f"{'='*80}")
    
    # Method success rates
    method_success = {
        'LASSCF': sum(1 for r in all_results if r['lasscf_energy'] is not None),
        'CASCI': sum(1 for r in all_results if r['casci_energy'] is not None),
        'LASSIrq': sum(1 for r in all_results if r['lassirq_energy'] is not None),
        'LASSIrq-VQE': sum(1 for r in all_results if r['lassirq_vqe_energy'] is not None)
    }
    
    total_tested = len(all_results)
    print(f"Total molecules tested: {total_tested}")
    print(f"\nMethod success rates:")
    for method, count in method_success.items():
        print(f"  {method:<12}: {count}/{total_tested} ({count/total_tested*100:.1f}%)")
    
    return all_results

if __name__ == "__main__":
    results = main()