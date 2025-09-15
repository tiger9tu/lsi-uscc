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


name = 'Stilbene-001'

ncas_sub = (4, 2, 4)
nelec_sub = ((2, 2), (1, 1), (2, 2))
frag_atom_list = ((1,2,3,4,5,6,15,16,17,18,19), (0,7,14,20), (8,9,10,11,12,13,21,22,23,24,25))
        
        # Fragment spin orbital indices (after LASSCF orbital ordering)
frag_spin_orbs = {
            0: (0, 1, 2, 3, 10, 11, 12, 13),    # Fragment 0: Phenyl ring 1
            1: (4, 5, 14, 15),                  # Fragment 1: Vinyl bridge
            2: (6, 7, 8, 9, 16, 17, 18, 19)    # Fragment 2: Phenyl ring 2
        }
frag_pairs = ((0, 1), (1, 2),(0,2))  # Fragment pairs for NOCI


geom = """C 0.637297 1.609894 0.271644
C 1.671915 0.637125 0.095037
C 1.555787 -0.429383 -0.818561
C 2.601892 -1.301075 -1.027366
C 3.804515 -1.141541 -0.343657
C 3.944438 -0.092275 0.554759
C 2.896357 0.779603 0.774827
C -0.637289 1.610117 -0.271666
C -1.671947 0.637404 -0.094976
C -1.556248 -0.428352 0.819552
C -2.602303 -1.300124 1.028279
C -3.804491 -1.141341 0.343632
C -3.944028 -0.092767 -0.555655
C -2.895976 0.77916 -0.775661
H 0.902253 2.448365 0.916227
H 0.632452 -0.54494 -1.373252
H 2.490883 -2.108975 -1.73971
H 4.627795 -1.820335 -0.523243
H 4.877985 0.045 1.08573
H 3.011386 1.601166 1.472498
H -0.901646 2.447808 -0.917488
H -0.633279 -0.543245 1.374995
H -2.491625 -2.107457 1.741317
H -4.627761 -1.820157 0.52318
H -4.877252 0.043926 -1.087344
H -3.010699 1.600174 -1.474031
"""

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