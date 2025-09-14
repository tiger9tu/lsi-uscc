#!/usr/bin/env python3
"""
Demo energy comparison for H6 6-31G to show methodology before stilbene
"""
import sys
sys.path.append('/home/jinx/repo/qchem/las-uscc-noci-bot/working')

from comprehensive_energy_comparison import compare_single_molecule
from lassi_instances_lassirq import molecular_configs

# Find H6 6-31G configuration
h6_config = None
for config in molecular_configs:
    if config['name'] == 'H6_631G':
        h6_config = config
        break

if h6_config is None:
    print("Error: Could not find H6_631G configuration")
    sys.exit(1)

print(f"Running energy comparison demo for {h6_config['name']}")
print(f"This demonstrates the methodology before running stilbene 6-31G")
result = compare_single_molecule(h6_config)

if result:
    print("\n" + "="*60)
    print("DEMO RESULTS SUMMARY FOR H6_631G")
    print("="*60)
    print(f"CASCI energy:     {result['casci_energy']:.10f} hartree" if result['casci_energy'] else "CASCI: Failed")
    print(f"LASSCF energy:    {result['lasscf_energy']:.10f} hartree" if result['lasscf_energy'] else "LASSCF: Failed")
    print(f"LASSIrq energy:   {result['lassirq_energy']:.10f} hartree" if result['lassirq_energy'] else "LASSIrq: Failed")
    print(f"LASSIrq-VQE energy: {result['lassirq_vqe_energy']:.10f} hartree" if result['lassirq_vqe_energy'] else "LASSIrq-VQE: Failed")
    
    if all(x is not None for x in [result['casci_energy'], result['lassirq_vqe_energy'], result['lassirq_energy'], result['lasscf_energy']]):
        print("\nEnergy Ordering Check:")
        energies = {
            'CASCI': result['casci_energy'],
            'LASSIrq-VQE': result['lassirq_vqe_energy'],
            'LASSIrq': result['lassirq_energy'],
            'LASSCF': result['lasscf_energy']
        }
        
        methods_order = ['CASCI', 'LASSIrq-VQE', 'LASSIrq', 'LASSCF']
        actual_energies = [energies[method] for method in methods_order]
        
        if all(actual_energies[i] <= actual_energies[i+1] for i in range(len(actual_energies)-1)):
            print("✓ Energy ordering is correct: CASCI ≤ LASSIrq-VQE ≤ LASSIrq ≤ LASSCF")
        else:
            print("✗ Energy ordering violation detected!")
            
        print("\nRelative energies (mEh vs CASCI):")
        ref_energy = result['casci_energy']
        for method in methods_order:
            rel_energy = (energies[method] - ref_energy) * 1000
            print(f"{method}: {rel_energy:+8.3f} mEh")
    
    print("\nThis methodology will be applied to stilbene 6-31G next")
else:
    print("Demo calculation failed")