#!/usr/bin/env python3
"""
Comprehensive Energy Comparison: LASSCF vs LASSIrq-VQE vs LASSIrq vs CASCI

This script runs all four methods on all molecular instances and compares their energies.
"""

import numpy as np
import time
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSIrq

from lassi_instances_lassirq import molecular_configs, setup_molecule

from mrh.exploratory.citools import grad, lasci_ominus1, fockspace
from mrh.exploratory.unitary_cc import lasuccsd
from scipy.linalg import eigh

def calculate_casci_energy(mol, mf, ncas_total, nelec_total, mo_coeff):
    """Calculate CASCI reference energy"""
    try:
        mc_casci = mcscf.CASCI(mf, ncas_total, nelec_total).set(
            fcisolver=csf_solver(mol, smult=1))
        mc_casci.kernel(mo_coeff)
        return mc_casci.e_tot if mc_casci.converged else None
    except Exception as e:
        print(f"    CASCI failed: {e}")
        return None

def calculate_lasscf_energy(mol, mf, mol_config):
    """Calculate LASSCF energy"""
    try:
        las = LASSCF(mf, mol_config['ncas'], mol_config['nelecas'], 
                     spin_sub=mol_config['spinsub'])
        mo_loc = las.localize_init_guess(mol_config['frag_atom_list'], mf.mo_coeff)
        las.kernel(mo_loc)
        return las.e_tot if las.converged else None, las
    except Exception as e:
        print(f"    LASSCF failed: {e}")
        return None, None

def calculate_lassirq_energy(las, r=1, q=1):
    """Calculate LASSIrq energy"""
    try:
        lsi = LASSIrq(las, r=r, q=q)
        e_roots, si_rq = lsi.kernel()
        return e_roots[0], lsi  # Ground state energy
    except Exception as e:
        print(f"    LASSIrq failed: {e}")
        return None, None

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

def calculate_lassirq_vqe_energy(mol, mf, las, lsi, max_cycle=3):
    """Calculate LASSIrq-VQE energy using the corrected implementation from mylassi.py"""
    try:
        # Get electron configurations
        nelec_fr = lsi.get_nelec_frs()
        ncore, ncas = las.ncore, las.ncas
        nelecas = sum(las.nelecas_sub)
        mo_coeff = las.mo_coeff
        
        # VQE optimization for each CT state
        nstates = len(lsi.ci[0])
        psis = []
        vqe_individual_energies = []  # Store individual VQE energies
        
        print(f"    Building VQE wavefunctions for {nstates} states...")
        
        for i in range(nstates):
            ci = [[lsi.ci[fragj][i]] for fragj in range(len(lsi.ci))]
            nelec_sub = [nelec_fr[fragj][i] for fragj in range(len(nelec_fr))]
            
            # Temporary LASSCF for gradient
            tmplas = LASSCF(mf, las.ncas_sub, nelec_sub)
            tmplas.mo_coeff = mo_coeff
            tmplas.ci = ci
            all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(tmplas, 0.00001)
            
            # VQE solver setup
            nelecas_sub = [sum(nelec) for nelec in nelec_sub]
            mc_uscc = mcscf.CASCI(mf, ncas, nelecas)
            mc_uscc.mo_coeff = mo_coeff
            mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
            mc_uscc.fcisolver.norb_f = las.ncas_sub
            lasci_ominus1.GLOBAL_MAX_CYCLE = max_cycle
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
        
        # Find lowest individual VQE energy before state interaction
        lowest_individual_vqe = min(vqe_individual_energies)
        print(f"    Lowest individual VQE energy: {lowest_individual_vqe:.10f} hartree")
        
        # Solve generalized eigenvalue problem
        eigvals, eigvecs = eigh(H, S)
        final_vqe_energy = eigvals[0]
        
        # State interaction improvement
        vqe_si_improvement = final_vqe_energy - lowest_individual_vqe
        print(f"    Final LASSIrq-VQE energy (after state interaction): {final_vqe_energy:.10f} hartree")
        print(f"    State interaction improvement: {vqe_si_improvement:+.10f} hartree")
        
        return final_vqe_energy, lowest_individual_vqe
        
    except Exception as e:
        print(f"    LASSIrq-VQE failed: {e}")
        return None, None

def compare_single_molecule(mol_config, max_vqe_cycle=3):
    """Compare all four methods for a single molecule"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {mol_config['name']}")
    print(f"{'='*60}")
    
    results = {
        'name': mol_config['name'],
        'ncas': mol_config['ncas'],
        'nelecas': mol_config['nelecas'],
        'lasscf_energy': None,
        'casci_energy': None,
        'lassirq_energy': None,
        'lassirq_vqe_energy': None,
        'lowest_individual_vqe': None,
        'lasscf_time': None,
        'casci_time': None,
        'lassirq_time': None,
        'lassirq_vqe_time': None,
        'notes': []
    }
    
    # Setup molecule
    mol, mf = setup_molecule(mol_config)
    ncas_total = sum(mol_config['ncas'])
    nelec_total = sum(mol_config['nelecas'])
    
    print(f"System: {ncas_total} orbitals, {nelec_total} electrons in {len(mol_config['ncas'])} fragments")
    print(f"RHF energy: {mf.e_tot:.10f} hartree")
    
    # 1. LASSCF calculation
    print(f"\n1. Running LASSCF...")
    start_time = time.time()
    lasscf_energy, las = calculate_lasscf_energy(mol, mf, mol_config)
    results['lasscf_time'] = time.time() - start_time
    
    if lasscf_energy is not None:
        results['lasscf_energy'] = lasscf_energy
        print(f"   LASSCF energy: {lasscf_energy:.10f} hartree")
    else:
        print(f"   LASSCF failed")
        results['notes'].append("LASSCF_FAILED")
        return results
    
    # 2. CASCI calculation
    print(f"2. Running CASCI reference...")
    start_time = time.time()
    casci_energy = calculate_casci_energy(mol, mf, ncas_total, nelec_total, las.mo_coeff)
    results['casci_time'] = time.time() - start_time
    
    if casci_energy is not None:
        results['casci_energy'] = casci_energy
        print(f"   CASCI energy: {casci_energy:.10f} hartree")
    else:
        print(f"   CASCI failed")
        results['notes'].append("CASCI_FAILED")
    
    # 3. LASSIrq calculation
    print(f"3. Running LASSIrq[1,1]...")
    start_time = time.time()
    lassirq_energy, lsi = calculate_lassirq_energy(las, r=1, q=1)
    results['lassirq_time'] = time.time() - start_time
    
    if lassirq_energy is not None:
        results['lassirq_energy'] = lassirq_energy
        print(f"   LASSIrq energy: {lassirq_energy:.10f} hartree")
    else:
        print(f"   LASSIrq failed")
        results['notes'].append("LASSIRQ_FAILED")
    
    # 4. LASSIrq-VQE calculation (requires successful LASSIrq)
    if lassirq_energy is not None and lsi is not None:
        print(f"4. Running LASSIrq-VQE (max_cycle={max_vqe_cycle})...")
        start_time = time.time()
        lassirq_vqe_result = calculate_lassirq_vqe_energy(mol, mf, las, lsi, max_cycle=max_vqe_cycle)
        results['lassirq_vqe_time'] = time.time() - start_time
        
        if lassirq_vqe_result[0] is not None:
            results['lassirq_vqe_energy'] = lassirq_vqe_result[0]
            results['lowest_individual_vqe'] = lassirq_vqe_result[1]
            print(f"   LASSIrq-VQE energy: {lassirq_vqe_result[0]:.10f} hartree")
        else:
            print(f"   LASSIrq-VQE failed")
            results['notes'].append("LASSIRQ_VQE_FAILED")
    else:
        print(f"4. Skipping LASSIrq-VQE (LASSIrq failed)")
        results['notes'].append("LASSIRQ_VQE_SKIPPED")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print(f"ENERGY SUMMARY FOR {mol_config['name']}")
    print(f"{'='*60}")
    energies = []
    labels = []
    
    if results['casci_energy'] is not None:
        energies.append(results['casci_energy'])
        labels.append("CASCI (lower bound)")
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
        if results['lowest_individual_vqe'] is not None:
            print(f"  Lowest individual VQE:  {results['lowest_individual_vqe']:15.10f} hartree")
            vqe_si_improvement = results['lassirq_vqe_energy'] - results['lowest_individual_vqe']
            print(f"  State interaction:      {vqe_si_improvement:+15.10f} hartree")
        if results['lassirq_energy'] is not None:
            diff = results['lassirq_vqe_energy'] - results['lassirq_energy']
            print(f"  (vs LASSIrq:            {diff:+15.10f} hartree)")
    
    # Energy ordering analysis
    if len(energies) >= 2:
        sorted_indices = np.argsort(energies)
        print(f"\nEnergy ordering (lowest to highest):")
        for i, idx in enumerate(sorted_indices):
            print(f"  {i+1}. {labels[idx]:15s} {energies[idx]:15.10f} hartree")
    
    return results

def main():
    """Run comprehensive comparison on all molecular instances"""
    print("=" * 80)
    print("COMPREHENSIVE ENERGY COMPARISON")
    print("LASSCF vs LASSIrq-VQE vs LASSIrq vs CASCI")
    print("=" * 80)
    
    all_results = []
    successful_calcs = 0
    total_calcs = len(molecular_configs)
    
    # Test on first few molecular configurations for validation
    test_configs = molecular_configs[:4]  # Test first 4 molecules
    for i, mol_config in enumerate(test_configs):
        print(f"\nProgress: {i+1}/{len(test_configs)} molecules")
        
        try:
            results = compare_single_molecule(mol_config, max_vqe_cycle=2)
            all_results.append(results)
            
            # Count successful calculations
            success_count = sum([
                results['lasscf_energy'] is not None,
                results['casci_energy'] is not None,
                results['lassirq_energy'] is not None,
                results['lassirq_vqe_energy'] is not None
            ])
            
            if success_count >= 2:
                successful_calcs += 1
                
        except Exception as e:
            print(f"ERROR: Failed to process {mol_config['name']}: {e}")
            continue
    
    # Create summary table
    print(f"\n{'='*80}")
    print("FINAL SUMMARY TABLE")
    print(f"{'='*80}")
    
    # Print table header
    header = f"{'Molecule':<15} {'Frags':<5} {'NCAS':<5} {'NELEC':<5} {'LASSCF':<12} {'CASCI':<12} {'LASSIrq':<12} {'LASSIrq-VQE':<12} {'Status':<15}"
    print(header)
    print("-" * len(header))
    
    for result in all_results:
        row = f"{result['name']:<15} "
        row += f"{len(result['ncas']):<5} "
        row += f"{sum(result['ncas']):<5} "
        row += f"{sum(result['nelecas']):<5} "
        row += f"{result['lasscf_energy']:.6f}" if result['lasscf_energy'] else "FAILED      "
        row += " "
        row += f"{result['casci_energy']:.6f}" if result['casci_energy'] else "FAILED      "
        row += " "
        row += f"{result['lassirq_energy']:.6f}" if result['lassirq_energy'] else "FAILED      "
        row += " "
        row += f"{result['lassirq_vqe_energy']:.6f}" if result['lassirq_vqe_energy'] else "FAILED      "
        row += " "
        row += "; ".join(result['notes']) if result['notes'] else "SUCCESS"
        print(row)
    
    # Energy differences analysis
    print(f"\n{'='*80}")
    print("ENERGY DIFFERENCES ANALYSIS")
    print(f"{'='*80}")
    
    successful_results = [r for r in all_results if all([
        r['lasscf_energy'] is not None,
        r['lassirq_energy'] is not None,
        r['casci_energy'] is not None
    ])]
    
    if successful_results:
        print(f"\nAnalysis for {len(successful_results)} molecules with all methods successful:")
        print(f"{'Molecule':<15} {'LASSIrq-LASSCF':<15} {'LASSCF-CASCI':<15} {'LASSIrq-CASCI':<15}")
        print("-" * 65)
        
        lassirq_lasscf_diffs = []
        lasscf_casci_diffs = []
        lassirq_casci_diffs = []
        
        for r in successful_results:
            lassirq_lasscf = r['lassirq_energy'] - r['lasscf_energy']
            lasscf_casci = r['lasscf_energy'] - r['casci_energy'] 
            lassirq_casci = r['lassirq_energy'] - r['casci_energy']
            
            print(f"{r['name']:<15} {lassirq_lasscf:+14.8f} {lasscf_casci:+14.8f} {lassirq_casci:+14.8f}")
            
            lassirq_lasscf_diffs.append(lassirq_lasscf)
            lasscf_casci_diffs.append(lasscf_casci)
            lassirq_casci_diffs.append(lassirq_casci)
        
        print("-" * 65)
        print(f"{'AVERAGE':<15} {np.mean(lassirq_lasscf_diffs):+14.8f} {np.mean(lasscf_casci_diffs):+14.8f} {np.mean(lassirq_casci_diffs):+14.8f}")
        print(f"{'STD DEV':<15} {np.std(lassirq_lasscf_diffs):14.8f} {np.std(lasscf_casci_diffs):14.8f} {np.std(lassirq_casci_diffs):14.8f}")
    
    print(f"\n{'='*80}")
    print(f"CALCULATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total molecules tested: {total_calcs}")
    print(f"Molecules with ≥2 successful methods: {successful_calcs}")
    print(f"Success rate: {successful_calcs/total_calcs*100:.1f}%")
    
    # Method success rates
    method_success = {
        'LASSCF': sum(1 for r in all_results if r['lasscf_energy'] is not None),
        'CASCI': sum(1 for r in all_results if r['casci_energy'] is not None),
        'LASSIrq': sum(1 for r in all_results if r['lassirq_energy'] is not None),
        'LASSIrq-VQE': sum(1 for r in all_results if r['lassirq_vqe_energy'] is not None)
    }
    
    print(f"\nMethod success rates:")
    for method, count in method_success.items():
        print(f"  {method:<12}: {count}/{total_calcs} ({count/total_calcs*100:.1f}%)")
    
    return all_results

if __name__ == "__main__":
    results = main()