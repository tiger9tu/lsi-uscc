#!/usr/bin/env python3
"""
C6 (6-31G) - 4-Parameter Combination Test
Tests 4 combinations: (thres=0.01, maxcycle=10), (thres=0.001, maxcycle=10), 
                     (thres=0.01, maxcycle=50), (thres=0.001, maxcycle=50)
"""

import sys
import os


from energy_comparison import EnergyComparison
import time

name = 'C6 (6-31G) - 4-Parameter Combination Test'
ncas_sub = (2, 2, 2)
nelec_sub = ((1, 1), (1, 1), (1, 1))
frag_atom_list = ((0, 2), (10, 11), (3, 1))  # C6 fragment atoms

# Fragment spin orbital indices (after LASSCF orbital ordering)
frag_spin_orbs = {
    0: (0, 1, 6, 7),    # Fragment 0 spin orbitals
    1: (2, 3, 8, 9),    # Fragment 1 spin orbitals  
    2: (4, 5, 10, 11)   # Fragment 2 spin orbitals
}

frag_pairs = ((0, 1), (1, 2), (0, 2))  # Fragment pairs for NOCI

geom = """C -3.075055 0.167498 0.000000
C 3.075055 -0.167498 0.000000
C -1.867388 -0.418911 0.000000
C 1.867388 0.418911 0.000000
H -3.180907 1.249260 0.000000
H 3.180907 -1.249260 0.000000
H -3.991128 -0.413469 0.000000
H 3.991128 0.413469 0.000000
H -1.805249 -1.507080 0.000000
H 1.805249 1.507080 0.000000
C -0.606997 0.297372 0.000000
C 0.606997 -0.297372 0.000000
H -0.661631 1.386399 0.000000
H 0.661631 -1.386399 0.000000"""

def main():
    print('='*80)
    print('Testing: (thres=0.01, maxcycle=10), (thres=0.001, maxcycle=10)')
    print('         (thres=0.01, maxcycle=50), (thres=0.001, maxcycle=50)')
    print('='*80)

    # Define test parameter combinations
    test_params = [
        {'thres': 0.01, 'maxcycle': 10},
        {'thres': 0.001, 'maxcycle': 10},
        {'thres': 0.01, 'maxcycle': 50},
        {'thres': 0.001, 'maxcycle': 50}
    ]
    
    # Results storage
    all_results = []
    
    # Use while loop to iterate through test cases
    i = 0
    while i < len(test_params):
        params = test_params[i]
        thres = params['thres']
        maxcycle = params['maxcycle']
        
        print(f'\n' + '='*70)
        print(f'TEST CASE {i+1}: GRADIENT THRESHOLD = {thres}, VQE MAX_CYCLES = {maxcycle}')
        print('='*70)
        
        start_time = time.time()
        
        try:
            comparison = EnergyComparison(
                                geom=geom,
                ncas_sub=ncas_sub,
                nelec_sub=nelec_sub,
                frag_atom_list=frag_atom_list,
                frag_spin_orbs=frag_spin_orbs,
                frag_pairs=frag_pairs,
                basis='6-31g',
                gradient_threshold=thres,
                vqe_max_cycles=maxcycle,
                                verbose=1,
            )
            
            results = comparison.run_all_methods()
            comparison.print_comparison_summary()
            
            calc_time = time.time() - start_time
            print(f'\nTotal time for thres={thres}, maxcycle={maxcycle}: {calc_time:.2f} seconds')
            
            # Store results with parameters
            test_result = {
                'thres': thres,
                'maxcycle': maxcycle,
                'results': results,
                'total_time': calc_time
            }
            all_results.append(test_result)
            
        except Exception as e:
            print(f'ERROR in test case {i+1} (thres={thres}, maxcycle={maxcycle}): {e}')
            import traceback
            traceback.print_exc()
            
            # Store error result
            test_result = {
                'thres': thres,
                'maxcycle': maxcycle,
                'results': None,
                'total_time': time.time() - start_time,
                'error': str(e)
            }
            all_results.append(test_result)
        
        i += 1
    
    # Print summary of all test cases
    print('\n' + '='*100)
    print('SUMMARY OF ALL 4 TEST CASES - ', name)
    print('='*100)
    
    print(f"{'Case':<6} {'Thres':<8} {'MaxCyc':<8} {'Time(s)':<8} {'Status':<10} {'Methods Completed':<20}")
    print('-' * 100)
    
    case_num = 1
    while case_num <= len(all_results):
        result = all_results[case_num - 1]
        thres = result['thres']
        maxcycle = result['maxcycle']
        total_time = result['total_time']
        
        if 'error' in result:
            status = "ERROR"
            methods_completed = "0/5"
        elif result['results'] is not None:
            status = "SUCCESS"
            methods_completed = f"{len(result['results'])}/5"
        else:
            status = "UNKNOWN"
            methods_completed = "?/5"
        
        print(f"{case_num:<6} {thres:<8} {maxcycle:<8} {total_time:<8.2f} {status:<10} {methods_completed:<20}")
        case_num += 1
    
    print('\n' + '='*80)
    print(name, '4-PARAMETER COMPARISON COMPLETE')
    print('='*80)
    
    return all_results

if __name__ == "__main__":
    main()