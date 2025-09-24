#!/usr/bin/env python3
"""
C6 Parameter Study: Testing different excitation thresholds and VQE cycles
"""

import sys
import time
import numpy as np
import random
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from pyscf import gto, scf, mcscf
from scipy.linalg import eigh

# Import our LAS-VQE-NOSI implementation
from las_vqe_nosi import LASVQENOSI
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.fci import csf_solver
from mrh.exploratory.citools import grad
from mrh.exploratory.unitary_cc import lasuccsd

@dataclass
class ComparisonResult:
    """Container for energy comparison results"""
    method_name: str
    energy: float
    n_excitations: int = 0
    n_states: int = 0
    H_matrix: np.ndarray = None
    S_matrix: np.ndarray = None
    min_diag_H: float = None
    calculation_time: float = 0.0
    converged: bool = False
    additional_info: Dict[str, Any] = None

def run_parameter_combination(excitation_threshold, vqe_max_cycle, output_file):
    """Run C6 comparison with specific parameters"""
    
    print(f"\n{'='*80}")
    print(f"Running C6 Parameter Study: threshold={excitation_threshold}, max_cycle={vqe_max_cycle}")
    print(f"{'='*80}")
    
    # C6 molecule geometry - polyene chain
    mol_str = '''
    C      0.000000    0.000000    0.000000
    C      1.339000    0.000000    0.000000
    C      2.008500    1.217500    0.000000
    C      3.347500    1.217500    0.000000
    C      4.017000    2.435000    0.000000
    C      5.356000    2.435000    0.000000
    H     -0.544500   -0.938500    0.000000
    H     -0.544500    0.938500    0.000000
    H      1.883500   -0.938500    0.000000
    H      1.464000    2.156000    0.000000
    H      3.892000    0.279000    0.000000
    H      3.472500    3.373500    0.000000
    H      5.900500    1.496500    0.000000
    H      5.900500    3.373500    0.000000
    '''
    
    # Initialize molecule
    mol = gto.Mole()
    mol.atom = mol_str
    mol.basis = 'sto-3g'
    mol.symmetry = False
    mol.verbose = 0
    mol.build()
    
    # Results storage
    results = {}
    energy_data = []
    
    print("Molecule built successfully")
    print(f"Number of atoms: {mol.natm}")
    print(f"Number of electrons: {mol.nelectron}")
    print(f"Number of orbitals: {mol.nao}")
    
    # Fragment definition for C6
    fragment_atoms = [[0, 2], [10, 11], [3, 1]]
    fragment_spin_orbs = [[0, 1, 6, 7], [2, 3, 8, 9], [4, 5, 10, 11]]
    fragment_charges = [0, 0, 0]
    fragment_spins = [0, 0, 0]  # Singlet ground state
    
    start_time = time.time()
    
    # 1. CASCI Reference
    print("\n" + "="*50)
    print("1. Running CASCI Reference")
    print("="*50)
    
    try:
        casci_start = time.time()
        
        # Mean-field calculation
        mf = scf.RHF(mol)
        mf.kernel()
        
        # CASCI calculation  
        mc_casci = mcscf.CASCI(mf, 12, 12)
        mc_casci.kernel()
        
        casci_time = time.time() - casci_start
        casci_energy = mc_casci.e_tot
        
        results['CASCI'] = {
            'energy': casci_energy,
            'time': casci_time,
            'converged': mc_casci.converged,
            'method': 'CASCI Reference',
            'n_excitations': 0,
            'n_states': 1
        }
        
        print(f"CASCI Energy: {casci_energy:.10f} hartree")
        print(f"CASCI Time: {casci_time:.2f} seconds")
        print(f"CASCI Converged: {mc_casci.converged}")
        
    except Exception as e:
        print(f"CASCI calculation failed: {e}")
        casci_energy = None
    
    # 2. LASSCF
    print("\n" + "="*50)
    print("2. Running LASSCF")
    print("="*50)
    
    try:
        lasscf_start = time.time()
        
        # LASSCF calculation using LASVQENOSI
        # Fragment configuration for C6: 3 fragments with 2 electrons and 2 orbitals each
        ncas_sub = (2, 2, 2)  # 2 orbitals per fragment
        nelec_sub = ((1, 1), (1, 1), (1, 1))  # (alpha, beta) electrons per fragment
        
        las_calc = LASVQENOSI(
            mol=mol,
            ncas_sub=ncas_sub,
            nelec_sub=nelec_sub,
            basis='sto-3g',
            frag_atom_list=tuple([tuple(frag) for frag in fragment_atoms]),
            gradient_threshold=excitation_threshold,
            vqe_max_cycles=vqe_max_cycle
        )
        
        # Setup molecule and perform LASSCF
        las_calc.setup_molecule()
        las_calc.perform_lasscf()
        las = las_calc.las
        
        print(f"LASSCF Energy: {las_calc.result.lasscf_energy:.10f} hartree")
        print(f"LASSCF Time: Included in total time")
        print(f"LASSCF Converged: {las_calc.result.lasscf_converged}")
        
        lasscf_time = time.time() - lasscf_start
        lasscf_energy = las.e_tot
        
        results['LASSCF'] = {
            'energy': las_calc.result.lasscf_energy,
            'time': lasscf_time,
            'converged': las_calc.result.lasscf_converged,
            'method': 'LASSCF',
            'n_excitations': 0,
            'n_states': 1
        }
        
        print(f"LASSCF Energy: {las_calc.result.lasscf_energy:.10f} hartree")
        print(f"LASSCF Time: {lasscf_time:.2f} seconds")
        print(f"LASSCF Converged: {las_calc.result.lasscf_converged}")
        
    except Exception as e:
        print(f"LASSCF calculation failed: {e}")
        las = None
        lasscf_energy = None
    
    if las is None:
        print("ERROR: LASSCF initialization failed - cannot proceed with VQE calculations")
        return results
    
    if not las.converged:
        print("WARNING: LASSCF not converged but proceeding with VQE calculations...")
    
    # 3. LAS-VQE (All Excitations)
    print("\n" + "="*50)
    print("3. Running LAS-VQE (All Excitations)")
    print("="*50)
    
    try:
        las_vqe_start = time.time()
        
        # Use the same LASSCF object from above
        # Get default excitation sets and merge them for all-excitations approach
        default_sets = las_calc.get_default_excitation_sets()
        
        # Merge all excitations into single set
        all_doubles = []
        all_singles = []
        for doubles, singles in default_sets:
            all_doubles.extend(doubles)
            all_singles.extend(singles)
        
        # Remove duplicates
        all_doubles = list(set(all_doubles))
        all_singles = list(set(all_singles))
        n_total_excitations = len(all_doubles) + len(all_singles)
        
        print(f"Total excitations found: {n_total_excitations}")
        
        # Set up single state VQE with all excitations
        las_calc.excitation_parameter_sets = [(all_doubles, all_singles)]
        vqe_states = las_calc.perform_vqe_on_sets()
        
        las_vqe_time = time.time() - las_vqe_start
        las_vqe_energy = vqe_states[0].energy if vqe_states else None
        
        results['LAS_VQE_All'] = {
            'energy': las_vqe_energy,
            'time': las_vqe_time,
            'converged': vqe_states[0].converged if vqe_states else False,
            'method': 'LAS-VQE (All Excitations)',
            'n_excitations': n_total_excitations,
            'n_states': 1
        }
        
        print(f"LAS-VQE (All) Energy: {las_vqe_energy:.10f} hartree")
        print(f"LAS-VQE (All) Time: {las_vqe_time:.2f} seconds")
        print(f"LAS-VQE (All) Converged: {vqe_states[0].converged if vqe_states else False}")
        
    except Exception as e:
        print(f"LAS-VQE (All) calculation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. LAS-VQE-NOSI Protocol 1 (Fragment-based)
    print("\n" + "="*50)
    print("4. Running LAS-VQE-NOSI Protocol 1 (Fragment-based)")
    print("="*50)
    
    try:
        protocol1_start = time.time()
        
        # Create new instance for Protocol 1
        las_vqe_nosi = LASVQENOSI(
            mol=mol,
            ncas_sub=ncas_sub,
            nelec_sub=nelec_sub,
            basis='sto-3g',
            frag_atom_list=tuple([tuple(frag) for frag in fragment_atoms]),
            gradient_threshold=excitation_threshold,
            vqe_max_cycles=vqe_max_cycle
        )
        
        # Setup molecule and perform LASSCF
        las_vqe_nosi.setup_molecule()
        las_vqe_nosi.perform_lasscf()
        
        # Get default excitation sets as base for fragment selection
        default_sets = las_vqe_nosi.get_default_excitation_sets()
        
        # Merge all excitations
        all_doubles = []
        all_singles = []
        for doubles, singles in default_sets:
            all_doubles.extend(doubles)
            all_singles.extend(singles)
        
        print(f"Total excitations available: {len(all_doubles) + len(all_singles)}")
        
        # Fragment-based division - simplified approach
        # Divide excitations among fragment pairs
        fragment_pairs = [(0, 1), (1, 2), (0, 2)]
        excitation_sets = []
        
        # Divide doubles roughly equally among fragment pairs
        n_per_fragment = len(all_doubles) // len(fragment_pairs)
        
        for i, frag_pair in enumerate(fragment_pairs):
            start_idx = i * n_per_fragment
            if i == len(fragment_pairs) - 1:  # Last pair gets remainder
                end_idx = len(all_doubles)
            else:
                end_idx = (i + 1) * n_per_fragment
            
            pair_doubles = all_doubles[start_idx:end_idx]
            pair_singles = []  # Simplified - no singles
            
            if pair_doubles:
                excitation_sets.append((pair_doubles, pair_singles))
                print(f"Fragment pair {frag_pair}: {len(pair_doubles)} excitations")
        
        if not excitation_sets:
            print("No fragment excitations found, using minimal set")
            # Fallback to minimal set
            doubles = all_doubles[:3] if len(all_doubles) >= 3 else all_doubles
            singles = []
            excitation_sets = [(doubles, singles)]
        
        # Set excitation parameter sets
        las_vqe_nosi.excitation_parameter_sets = excitation_sets
        
        # Perform VQE on each set
        las_vqe_nosi.perform_vqe_on_sets()
        
        # Perform state interaction if multiple states
        if len(excitation_sets) > 1:
            las_vqe_nosi.perform_state_interaction()
        
        protocol1_time = time.time() - protocol1_start
        
        # Get results
        final_energy = las_vqe_nosi.result.ground_state_energy if las_vqe_nosi.result.ground_state_energy else las_vqe_nosi.vqe_states[0].energy
        individual_energies = [state.energy for state in las_vqe_nosi.vqe_states] if las_vqe_nosi.vqe_states else []
        n_total_excitations = sum(len(exc_set[0]) + len(exc_set[1]) for exc_set in excitation_sets)
        
        results['Protocol_1'] = {
            'energy': final_energy,
            'time': protocol1_time,
            'converged': any([state.converged for state in las_vqe_nosi.vqe_states]) if las_vqe_nosi.vqe_states else False,
            'method': 'LAS-VQE-NOSI Protocol 1',
            'n_excitations': n_total_excitations,
            'n_states': len(excitation_sets),
            'individual_energies': individual_energies,
            'h_matrix': las_vqe_nosi.result.hamiltonian_matrix.tolist() if las_vqe_nosi.result.hamiltonian_matrix is not None else None,
            's_matrix': las_vqe_nosi.result.overlap_matrix.tolist() if las_vqe_nosi.result.overlap_matrix is not None else None
        }
        
        print(f"Protocol 1 Final Energy: {final_energy:.10f} hartree")
        print(f"Protocol 1 Time: {protocol1_time:.2f} seconds")
        print(f"Protocol 1 States: {len(excitation_sets)}")
        
    except Exception as e:
        print(f"Protocol 1 calculation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. LAS-VQE-NOSI Protocol 2 (Random Division)
    print("\n" + "="*50)
    print("5. Running LAS-VQE-NOSI Protocol 2 (Random Division)")
    print("="*50)
    
    try:
        protocol2_start = time.time()
        
        # Create new instance for Protocol 2
        las_vqe_nosi2 = LASVQENOSI(
            mol=mol,
            ncas_sub=ncas_sub,
            nelec_sub=nelec_sub,
            basis='sto-3g',
            frag_atom_list=tuple([tuple(frag) for frag in fragment_atoms]),
            gradient_threshold=excitation_threshold,
            vqe_max_cycles=vqe_max_cycle
        )
        
        # Setup molecule and perform LASSCF
        las_vqe_nosi2.setup_molecule()
        las_vqe_nosi2.perform_lasscf()
        
        # Get default excitation sets
        default_sets2 = las_vqe_nosi2.get_default_excitation_sets()
        
        # Merge all excitations
        all_doubles = []
        all_singles = []
        for doubles, singles in default_sets2:
            all_doubles.extend(doubles)
            all_singles.extend(singles)
        
        print(f"Total excitations for Protocol 2: {len(all_doubles) + len(all_singles)}")
        
        # Random division into 3 sets - separate doubles and singles
        random.seed(42)
        random.shuffle(all_doubles)
        
        n_sets = 3
        doubles_per_set = len(all_doubles) // n_sets
        
        excitation_sets = []
        for i in range(n_sets):
            start_idx = i * doubles_per_set
            if i == n_sets - 1:  # Last set gets remainder
                end_idx = len(all_doubles)
            else:
                end_idx = (i + 1) * doubles_per_set
            
            set_doubles = all_doubles[start_idx:end_idx]
            set_singles = []  # Simplified - no singles
            
            if set_doubles:
                excitation_sets.append((set_doubles, set_singles))
                print(f"Random set {i+1}: {len(set_doubles)} excitations")
        
        if not excitation_sets:
            print("No excitations found for Protocol 2")
            return results
        
        # Set excitation parameter sets
        las_vqe_nosi2.excitation_parameter_sets = excitation_sets
        
        # Perform VQE on each set
        las_vqe_nosi2.perform_vqe_on_sets()
        
        # Perform state interaction
        if len(excitation_sets) > 1:
            las_vqe_nosi2.perform_state_interaction()
        
        protocol2_time = time.time() - protocol2_start
        
        # Get results
        final_energy = las_vqe_nosi2.result.ground_state_energy if las_vqe_nosi2.result.ground_state_energy else las_vqe_nosi2.vqe_states[0].energy
        individual_energies = [state.energy for state in las_vqe_nosi2.vqe_states] if las_vqe_nosi2.vqe_states else []
        n_total_excitations = sum(len(exc_set[0]) + len(exc_set[1]) for exc_set in excitation_sets)
        
        results['Protocol_2'] = {
            'energy': final_energy,
            'time': protocol2_time,
            'converged': any([state.converged for state in las_vqe_nosi2.vqe_states]) if las_vqe_nosi2.vqe_states else False,
            'method': 'LAS-VQE-NOSI Protocol 2',
            'n_excitations': n_total_excitations,
            'n_states': len(excitation_sets),
            'individual_energies': individual_energies,
            'h_matrix': las_vqe_nosi2.result.hamiltonian_matrix.tolist() if las_vqe_nosi2.result.hamiltonian_matrix is not None else None,
            's_matrix': las_vqe_nosi2.result.overlap_matrix.tolist() if las_vqe_nosi2.result.overlap_matrix is not None else None
        }
        
        print(f"Protocol 2 Final Energy: {final_energy:.10f} hartree")
        print(f"Protocol 2 Time: {protocol2_time:.2f} seconds")
        print(f"Protocol 2 States: {len(excitation_sets)}")
        
    except Exception as e:
        print(f"Protocol 2 calculation failed: {e}")
        import traceback
        traceback.print_exc()
    
    total_time = time.time() - start_time
    print(f"\nTotal calculation time: {total_time:.2f} seconds")
    
    # Write results to markdown file
    write_results_markdown(results, excitation_threshold, vqe_max_cycle, output_file, casci_energy)
    
    return results

def write_results_markdown(results, threshold, max_cycle, output_file, casci_reference):
    """Write results to markdown file"""
    
    content = f"""# C6 Energy Comparison - Threshold {threshold}, Max Cycle {max_cycle}

## Overview

**Molecule**: C6 polyene chain  
**Basis**: STO-3G  
**Excitation Threshold**: {threshold}  
**VQE Max Cycles**: {max_cycle}  
**Fragment Atoms**: (0,2), (10,11), (3,1)  
**Fragment Spin Orbitals**: Frag 0: (0,1,6,7), Frag 1: (2,3,8,9), Frag 2: (4,5,10,11)  

## Energy Results Summary

"""
    
    if casci_reference:
        content += f"**CASCI Reference Energy**: {casci_reference:.10f} hartree\n\n"
    
    content += "| Method | Energy (hartree) | ΔE (mEh) | N_excitations | N_states | Time(s) | Converged |\n"
    content += "|--------|------------------|----------|---------------|----------|---------|-----------|\n"
    
    # Sort results by energy (lowest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['energy'] if x[1]['energy'] is not None else float('inf'))
    
    for method_name, data in sorted_results:
        energy = data['energy']
        if energy is None:
            continue
            
        # Calculate energy difference from CASCI
        if casci_reference:
            delta_e = (energy - casci_reference) * 1000  # Convert to mEh
        else:
            delta_e = 0.0
        
        converged_symbol = "✓" if data.get('converged', False) else "✗"
        
        content += f"| **{data['method']}** | {energy:.10f} | {delta_e:.3f} | {data['n_excitations']} | {data['n_states']} | {data['time']:.2f} | {converged_symbol} |\n"
    
    # Detailed analysis for state interaction methods
    content += "\n## Detailed State Interaction Analysis\n\n"
    
    for method_name, data in results.items():
        if 'Protocol' in method_name and data.get('h_matrix') and data.get('s_matrix'):
            content += f"### {data['method']}\n\n"
            
            if data.get('individual_energies'):
                content += "**Individual VQE Energies Before State Interaction:**\n"
                for i, e in enumerate(data['individual_energies']):
                    content += f"- State {i+1}: {e:.8f} hartree ({data['n_excitations']//data['n_states']} excitations)\n"
                content += f"\n**Final Energy After State Interaction:** {data['energy']:.10f} hartree\n\n"
            
            if data.get('h_matrix'):
                content += "**Hamiltonian Matrix (hartree):**\n```\n"
                h_matrix = np.array(data['h_matrix'])
                for row in h_matrix:
                    content += "[" + "  ".join(f"{val:.8f}" for val in row) + "]\n"
                content += "```\n\n"
            
            if data.get('s_matrix'):
                content += "**Overlap Matrix:**\n```\n"
                s_matrix = np.array(data['s_matrix'])
                for row in s_matrix:
                    content += "[" + "  ".join(f"{val:.8f}" for val in row) + "]\n"
                content += "```\n\n"
            
            # Calculate state interaction benefit
            if data.get('individual_energies'):
                best_individual = min(data['individual_energies'])
                improvement = (best_individual - data['energy']) * 1000  # mEh
                content += f"**State Interaction Benefits:**\n"
                content += f"- **Energy Lowering**: {improvement:.2f} mEh improvement (from best individual {best_individual:.8f})\n\n"
    
    # Method performance analysis
    content += "## Method Performance Analysis\n\n"
    
    if casci_reference:
        content += f"### Energy Accuracy Ranking (vs CASCI: {casci_reference:.10f} hartree)\n"
        for i, (method_name, data) in enumerate(sorted_results, 1):
            if data['energy'] is not None:
                delta_e = (data['energy'] - casci_reference) * 1000
                content += f"{i}. **{data['method']}**: {data['energy']:.10f} hartree ({delta_e:.3f} mEh)\n"
    
    content += f"\n### Computational Efficiency\n"
    time_sorted = sorted(results.items(), key=lambda x: x[1]['time'])
    for i, (method_name, data) in enumerate(time_sorted, 1):
        content += f"{i}. **{data['method']}**: {data['time']:.2f}s\n"
    
    # Technical insights
    content += f"\n## Technical Insights\n\n"
    content += f"### Parameter Effects\n"
    content += f"- **Excitation Threshold**: {threshold}\n"
    content += f"  - Controls number of excitations included in VQE optimization\n"
    content += f"  - Lower threshold → more excitations → potentially better accuracy\n"
    content += f"- **VQE Max Cycles**: {max_cycle}\n"
    content += f"  - Controls VQE convergence thoroughness\n"
    content += f"  - More cycles → better convergence but higher computational cost\n\n"
    
    content += f"### Convergence Analysis\n"
    converged_methods = [data['method'] for data in results.values() if data.get('converged', False)]
    unconverged_methods = [data['method'] for data in results.values() if not data.get('converged', False)]
    
    if converged_methods:
        content += f"**Converged Methods**: {', '.join(converged_methods)}\n"
    if unconverged_methods:
        content += f"**Unconverged Methods**: {', '.join(unconverged_methods)}\n"
    
    content += f"\n## Conclusions\n\n"
    
    if len(sorted_results) > 0 and sorted_results[0][1]['energy'] is not None:
        best_method = sorted_results[0][1]['method']
        best_energy = sorted_results[0][1]['energy']
        content += f"**Best Method**: {best_method} with energy {best_energy:.10f} hartree\n\n"
    
    content += f"**Parameter Combination**: threshold={threshold}, max_cycle={max_cycle}\n"
    if casci_reference:
        content += f"**CASCI Reference**: {casci_reference:.10f} hartree\n"
    
    content += f"\nThis parameter combination demonstrates the trade-offs between excitation coverage (threshold) and optimization thoroughness (max_cycle) in LAS-VQE-NOSI calculations."
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"\nResults written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python c6_parameter_study.py <threshold> <max_cycle> <output_file>")
        sys.exit(1)
    
    threshold = float(sys.argv[1])
    max_cycle = int(sys.argv[2])
    output_file = sys.argv[3]
    
    run_parameter_combination(threshold, max_cycle, output_file)