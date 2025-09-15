#!/usr/bin/env python3
"""
Stilbene-60 - VQE MAX_CYCLES 10 vs 50 COMPARISON TEST
Gradient threshold: 0.01
"""

import sys
import os
sys.path.append('/home/jinx/repo/qchem/las-uscc-noci-bot/working/sept')

exec(open('./stilbene-60_energy_comparison.py').read())
import time

def main():
    print('='*80)
    print('STILBENE-60 - VQE MAX_CYCLES 10 vs 50 COMPARISON')
    print('Gradient threshold: 0.01')
    print('='*80)

    # Test case 1: max_cycles = 10
    print('\n' + '='*60)
    print('TEST CASE 1: VQE MAX_CYCLES = 10')
    print('='*60)
    start_time = time.time()

    try:
        comparison_10 = Stilbene60EnergyComparison(
            basis='sto-3g',
            gradient_threshold=0.01,
            vqe_max_cycles=10,
            verbose=1
        )
        results_10 = comparison_10.run_all_methods()
        comparison_10.print_comparison_summary()
        time_10 = time.time() - start_time
        print(f'\nTotal time for max_cycles=10: {time_10:.2f} seconds')
        
    except Exception as e:
        print(f'ERROR in max_cycles=10 test: {e}')
        import traceback
        traceback.print_exc()

    print('\n' + '='*60)
    print('TEST CASE 2: VQE MAX_CYCLES = 50')
    print('='*60)
    start_time = time.time()

    try:
        comparison_50 = Stilbene60EnergyComparison(
            basis='sto-3g',
            gradient_threshold=0.01,
            vqe_max_cycles=50,
            verbose=1
        )
        results_50 = comparison_50.run_all_methods()
        comparison_50.print_comparison_summary()
        time_50 = time.time() - start_time
        print(f'\nTotal time for max_cycles=50: {time_50:.2f} seconds')
        
    except Exception as e:
        print(f'ERROR in max_cycles=50 test: {e}')
        import traceback
        traceback.print_exc()

    print('\n' + '='*80)
    print('STILBENE-60 COMPARISON COMPLETE')
    print('='*80)

if __name__ == "__main__":
    main()