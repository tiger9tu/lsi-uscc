#!/usr/bin/env python3
"""
Run single-molecule comparison for Stilbene 6-31G
"""
import sys
sys.path.append('/home/jinx/repo/qchem/las-uscc-noci-bot/working')

from comprehensive_energy_comparison import run_single_molecule_comparison
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

print(f"Running energy comparison for {stilbene_config['name']}")
result = run_single_molecule_comparison(stilbene_config)

if result:
    print("\nResults saved successfully")
else:
    print("Calculation failed")