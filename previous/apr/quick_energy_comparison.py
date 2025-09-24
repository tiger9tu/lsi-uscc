#!/usr/bin/env python3
"""
Quick Energy Comparison: LASSCF vs LASSIrq vs CASCI

This script runs the three working methods on all molecular instances and compares their energies.
"""

import numpy as np
import time
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSIrq

from lassi_instances_lassirq import molecular_configs, setup_molecule

def calculate_all_energies(mol_config):
    """Calculate LASSCF, LASSIrq, and CASCI energies for a single molecule"""
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
        'num_lassirq_states': None
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
        
        if las.converged:
            results['lasscf_energy'] = las.e_tot
            print(f"   LASSCF energy: {las.e_tot:.10f} hartree")
        else:
            print(f"   LASSCF failed to converge")
            return results
        
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
        except Exception as e:
            print(f"   CASCI failed: {e}")
        
        # 3. LASSIrq calculation
        print(f"3. Running LASSIrq[1,1]...")
        try:
            lsi = LASSIrq(las, r=1, q=1)
            e_roots, si_rq = lsi.kernel()
            results['lassirq_energy'] = e_roots[0]
            results['num_lassirq_states'] = len(e_roots)
            print(f"   LASSIrq energy: {e_roots[0]:.10f} hartree")
            print(f"   Number of states: {len(e_roots)}")
        except Exception as e:
            print(f"   LASSIrq failed: {e}")
        
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
        
        # Energy ordering
        if len(energies) >= 2:
            sorted_indices = np.argsort(energies)
            print(f"\nEnergy ordering (lowest to highest):")
            for i, idx in enumerate(sorted_indices):
                print(f"  {i+1}. {labels[idx]:15s} {energies[idx]:15.10f} hartree")
        
        return results
        
    except Exception as e:
        print(f"ERROR: {e}")
        return results

def main():
    """Run comparison on all molecular instances"""
    print("=" * 80)
    print("ENERGY COMPARISON: LASSCF vs LASSIrq vs CASCI")
    print("=" * 80)
    
    all_results = []
    successful_calcs = 0
    total_calcs = len(molecular_configs)
    
    # Process all molecules
    for i, mol_config in enumerate(molecular_configs):
        print(f"\nProgress: {i+1}/{total_calcs} molecules")
        
        results = calculate_all_energies(mol_config)
        all_results.append(results)
        
        # Count as successful if at least LASSCF and one other method worked
        if (results['lasscf_energy'] is not None and 
            (results['casci_energy'] is not None or results['lassirq_energy'] is not None)):
            successful_calcs += 1
    
    # Summary table
    print(f"\n{'='*80}")
    print("FINAL SUMMARY TABLE")
    print(f"{'='*80}")
    
    header = f"{'Molecule':<15} {'Frags':<5} {'NCAS':<5} {'RHF':<12} {'LASSCF':<12} {'LASSIrq':<12} {'CASCI':<12} {'States':<7}"
    print(header)
    print("-" * len(header))
    
    for result in all_results:
        row = f"{result['name']:<15} "
        row += f"{len(result['ncas']):<5} "
        row += f"{sum(result['ncas']):<5} "
        row += f"{result['rhf_energy']:.6f}" if result['rhf_energy'] else "FAILED      "
        row += " "
        row += f"{result['lasscf_energy']:.6f}" if result['lasscf_energy'] else "FAILED      "
        row += " "
        row += f"{result['lassirq_energy']:.6f}" if result['lassirq_energy'] else "FAILED      "
        row += " "
        row += f"{result['casci_energy']:.6f}" if result['casci_energy'] else "FAILED      "
        row += " "
        row += f"{result['num_lassirq_states']:<7}" if result['num_lassirq_states'] else "N/A    "
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
        
        # Physical interpretation
        print(f"\n{'='*80}")
        print("PHYSICAL INTERPRETATION")
        print(f"{'='*80}")
        print("Expected energy ordering (lowest to highest):")
        print("1. CASCI (exact within active space) - LOWEST")  
        print("2. LASSIrq (includes charge transfer correlation)")
        print("3. LASSCF (mean-field treatment) - HIGHEST")
        print("")
        
        # Check if ordering is correct
        correct_ordering = 0
        for r in successful_results:
            if r['casci_energy'] <= r['lassirq_energy'] <= r['lasscf_energy']:
                correct_ordering += 1
        
        print(f"Molecules with correct energy ordering: {correct_ordering}/{len(successful_results)}")
        print(f"Success rate: {correct_ordering/len(successful_results)*100:.1f}%")
    
    print(f"\n{'='*80}")
    print(f"CALCULATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total molecules tested: {total_calcs}")
    print(f"Successful calculations: {successful_calcs}")
    print(f"Success rate: {successful_calcs/total_calcs*100:.1f}%")
    
    # Method success rates
    method_success = {
        'LASSCF': sum(1 for r in all_results if r['lasscf_energy'] is not None),
        'CASCI': sum(1 for r in all_results if r['casci_energy'] is not None),
        'LASSIrq': sum(1 for r in all_results if r['lassirq_energy'] is not None)
    }
    
    print(f"\nMethod success rates:")
    for method, count in method_success.items():
        print(f"  {method:<12}: {count}/{total_calcs} ({count/total_calcs*100:.1f}%)")
    
    return all_results

if __name__ == "__main__":
    results = main()