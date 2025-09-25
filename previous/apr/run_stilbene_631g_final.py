#!/usr/bin/env python3
"""
Run energy comparison for Stilbene 6-31G - Final Implementation
"""
import sys
sys.path.append('/home/jinx/repo/qchem/las-uscc-noci-bot/working')

from comprehensive_energy_comparison import compare_single_molecule
from lassi_instances_lassirq import molecular_configs

# Find stilbene 6-31G configuration
stilbene_config = None
for config in molecular_configs:
    if config['name'] == 'STIL_631G_90':
        stilbene_config = config
        break

if stilbene_config is None:
    print("Error: Could not find STIL_631G_90 configuration")
    print("Available configurations:")
    for config in molecular_configs:
        print(f"  - {config['name']}")
    sys.exit(1)

print(f"Running LASSIRQ-VQE energy comparison for {stilbene_config['name']}")
print(f"System: {sum(stilbene_config['ncas'])} orbitals, {sum(stilbene_config['nelecas'])} electrons in {len(stilbene_config['ncas'])} fragments")
print(f"Fragment configuration: {stilbene_config['ncas']}")

# Reduce VQE cycles to save time for demo
result = compare_single_molecule(stilbene_config, max_vqe_cycle=2)

if result:
    print("\n" + "="*70)
    print("FINAL ENERGY COMPARISON RESULTS FOR STILBENE 6-31G")
    print("="*70)
    
    # Print individual method energies
    methods_results = {
        'CASCI': result['casci_energy'],
        'LASSCF': result['lasscf_energy'],
        'LASSIrq': result['lassirq_energy'],
        'LASSIrq-VQE': result['lassirq_vqe_energy'],
    }
    
    print("Individual Method Energies:")
    print("-" * 40)
    for method, energy in methods_results.items():
        if energy is not None:
            print(f"{method:<15}: {energy:15.10f} hartree")
        else:
            print(f"{method:<15}: Failed")
    
    # Special extraction for lowest individual VQE energy
    if hasattr(result, 'lowest_vqe_individual'):
        print(f"{'Lowest VQE (before SI)':<15}: {result.lowest_vqe_individual:15.10f} hartree")
    
    # Energy comparison and ordering
    valid_energies = {k: v for k, v in methods_results.items() if v is not None}
    
    if len(valid_energies) >= 4:
        print("\nEnergy Ordering Analysis:")
        print("-" * 40)
        
        # Check expected ordering: CASCI ≤ LASSIrq-VQE ≤ LASSIrq ≤ LASSCF
        expected_order = ['CASCI', 'LASSIrq-VQE', 'LASSIrq', 'LASSCF']
        actual_energies = [valid_energies[method] for method in expected_order if method in valid_energies]
        
        if all(actual_energies[i] <= actual_energies[i+1] for i in range(len(actual_energies)-1)):
            print("✓ Energy ordering is CORRECT: CASCI ≤ LASSIrq-VQE ≤ LASSIrq ≤ LASSCF")
        else:
            print("✗ Energy ordering VIOLATION detected!")
            
        print("\nRelative Energies (mEh relative to CASCI):")
        print("-" * 50)
        ref_energy = valid_energies['CASCI']
        for method in expected_order:
            if method in valid_energies:
                rel_energy = (valid_energies[method] - ref_energy) * 1000
                print(f"{method:<15}: {rel_energy:+8.3f} mEh")
    
    # Additional VQE analysis
    if 'LASSIrq-VQE' in valid_energies and 'LASSIrq' in valid_energies:
        vqe_improvement = (valid_energies['LASSIrq'] - valid_energies['LASSIrq-VQE']) * 1000
        print(f"\nLASSIrq-VQE vs LASSIrq improvement: {vqe_improvement:.3f} mEh")
        
    if 'CASCI' in valid_energies and 'LASSIrq-VQE' in valid_energies:
        casci_gap = (valid_energies['LASSIrq-VQE'] - valid_energies['CASCI']) * 1000
        print(f"LASSIrq-VQE vs CASCI gap: {casci_gap:.3f} mEh")
    
    print("\n" + "="*70)
    print("CALCULATION COMPLETED SUCCESSFULLY")
    print("="*70)
    
else:
    print("ERROR: Stilbene 6-31G calculation failed!")
    sys.exit(1)