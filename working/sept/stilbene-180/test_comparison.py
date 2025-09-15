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


geom = """C       0.49510780     -0.45268682     -0.00006800
C       1.92255623     -0.18914892     -0.00006000
C       2.46957402      1.10064756     -0.00007300
C       3.83726147      1.29254348     -0.00006000
C       4.70216213      0.20499392     -0.00004000
C       4.18093133     -1.07917457     -0.00003800
C       2.81041788     -1.27088549     -0.00005500
C      -0.49510880      0.45268482      0.00043400
C      -1.92255723      0.18914892      0.00022900
C      -2.46957502     -1.10064756      0.00016900
C      -3.83726147     -1.29254348     -0.00006600
C      -4.70216113     -0.20499292     -0.00023000
C      -4.18093133      1.07917457     -0.00015300
C      -2.81041788      1.27088649      0.00008500
H       0.23519591     -1.50769340      0.00012100
H       1.81440028      1.96266822     -0.00010300
H       4.23644231      2.29925908     -0.00006600
H       5.77332670      0.36067086     -0.00002200
H       4.84437107     -1.93491023     -0.00003000
H       2.40613504     -2.27687709     -0.00006300
H      -0.23519491      1.50769040      0.00006500
H      -1.81439828     -1.96266722      0.00031000
H      -4.23644331     -2.29925808     -0.00010600
H      -5.77332570     -0.36066886     -0.00041100
H      -4.84437107      1.93491023     -0.00028300
H      -2.40613504      2.27687709      0.00014000
"""

def main():
    print('='*80)
    print('Testing: (thres=0.01, maxcycle=10), (thres=0.001, maxcycle=10)')
    print('         (thres=0.01, maxcycle=50), (thres=0.001, maxcycle=50)')
    print('='*80)

    # Define test parameter combinations
    test_params = [
        {'thres': 0.1, 'maxcycle': 1},
        # {'thres': 0.001, 'maxcycle': 10},
        # {'thres': 0.01, 'maxcycle': 50},
        # {'thres': 0.001, 'maxcycle': 50}
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