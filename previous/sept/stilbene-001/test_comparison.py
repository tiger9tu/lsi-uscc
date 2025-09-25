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


geom = """C      -0.66913973      1.85730726     -0.03805098
C      -1.61065036      0.73885071     -0.09574696
C      -1.35340246     -0.43288283     -0.81483568
C      -2.28594009     -1.45288042     -0.86682865
C      -3.49986561     -1.32918247     -0.20406092
C      -3.77981149     -0.16564793      0.49591780
C      -2.85072586      0.86034066      0.53755379
C       0.66914373      1.85730826      0.03787198
C       1.61065936      0.73880971      0.09552496
C       1.35328446     -0.43291083      0.81460268
C       2.28583609     -1.45287842      0.86681365
C       3.49987661     -1.32917547      0.20424692
C       3.77992149     -0.16567693     -0.49573780
C       2.85081786      0.86029966     -0.53757779
H      -1.14269154      2.83473987     -0.06049798
H      -0.41631383     -0.53217479     -1.34747946
H      -2.06882318     -2.34798206     -1.43597143
H      -4.22727632     -2.12958615     -0.24491490
H      -4.72859812     -0.05278198      1.00495860
H      -3.07755377      1.77276929      1.07689257
H       1.14262454      2.83473387      0.06205798
H       0.41611283     -0.53215779      1.34710046
H       2.06866718     -2.34795906      1.43596843
H       4.22730132     -2.12955815      0.24528090
H       4.72878912     -0.05283198     -1.00463060
H       3.07771977      1.77270129     -1.07692857
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