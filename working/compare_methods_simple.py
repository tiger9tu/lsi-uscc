#!/usr/bin/env python3
"""
Simplified comparison of LASSI and CASCI methods

This script focuses on comparing LASSI and CASCI for H4 and H6 molecules,
with detailed analysis of the energy differences and what they represent.
"""

import numpy as np
from pyscf import gto, scf, mcscf, lib
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSI
from mrh.my_pyscf.fci import csf_solver

# Molecular configurations
H4xyz = ''' H      0.000000000000   0.000000000000   0.000000000000
H      1.000000000000   0.000000000000   0.000000000000
H      0.273746762116   2.195450598147   0.100000000000
H      1.232912762116   1.895450598147  -0.100000000000
'''

H6xyz = ''' H      0.000000000000   0.000000000000   0.000000000000
H      1.000000000000   0.000000000000   0.000000000000
H      0.273746762116   2.195450598147   0.100000000000
H      1.232912762116   1.895450598147  -0.100000000000
H      0.507178110854   4.193780995243   0.049334760036
H      1.506140937609   3.988021397347  -0.049334760036
'''

def setup_molecule(xyz, basis='sto-3g'):
    """Setup molecule and HF reference"""
    mol = gto.M(atom=xyz, basis=basis, verbose=0, output=None)
    mf = scf.RHF(mol).run()
    return mol, mf

def run_casci(mol, mf, ncas, nelecas):
    """Run CASCI calculation"""
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.fcisolver = csf_solver(mol, smult=1)  # Singlet ground state
    mc.kernel()
    return mc.e_tot, mc.converged

def run_lassi_single_state(mol, mf, ncas_list, nelecas_list, spinsub_list, frag_atom_list):
    """Run LASSI calculation with single state for ground state comparison"""
    
    # Setup LASSCF
    las = LASSCF(mf, ncas_list, nelecas_list, spin_sub=spinsub_list)
    mo_loc = las.localize_init_guess(frag_atom_list, mf.mo_coeff)
    las.kernel(mo_loc)
    print("lasci energy =", las.e_tot)
    # Setup single-state LASSI for ground state
    las_sa = las.state_average([1.0, 0.0], 
                              spins=[[1,-1],[-1,1]], 
                              smults=[[2,2],[2,2]], 
                              charges=[[0,0],[0,0]])
    las_sa.lasci()
    
    # Create and run LASSI
    lsi = LASSI(las_sa)
    e_roots, si = lsi.kernel()
    
    return e_roots[0], las.e_tot, las.converged

def run_lassi_multi_state(mol, mf, ncas_list, nelecas_list, spinsub_list, frag_atom_list):
    """Run LASSI calculation with multiple states"""
    
    # Setup LASSCF
    las = LASSCF(mf, ncas_list, nelecas_list, spin_sub=spinsub_list)
    mo_loc = las.localize_init_guess(frag_atom_list, mf.mo_coeff)
    las.kernel(mo_loc)
    
    # Setup multi-state LASSI with various spin configurations
    nfrags = len(ncas_list)
    if nfrags == 2:
        weights = [0.6, 0.4]
        spins = [[1, 1], [1, -1]]
        smults = [[2, 2], [2, 2]]
        charges = [[0, 0], [0, 0]]
    elif nfrags == 3:
        weights = [0.5, 0.3, 0.2]  
        spins = [[1, 1, 1], [1, 1, -1], [-1, 1, 1]]
        smults = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
        charges = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    else:
        # Default single state
        weights = [1.0]
        spins = [[1]*nfrags]
        smults = [[2]*nfrags]
        charges = [[0]*nfrags]
    
    las_sa = las.state_average(weights, spins=spins, smults=smults, charges=charges)
    las_sa.lasci()
    
    # Create and run LASSI
    lsi = LASSI(las_sa)
    e_roots, si = lsi.kernel()
    
    return e_roots[0], e_roots, las_sa.nroots, las.converged

def compare_molecule(name, xyz, ncas_list, nelecas_list, spinsub_list, frag_atom_list):
    """Compare CASCI vs LASSI for a single molecule"""
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS FOR {name}")
    print(f"{'='*70}")
    
    # Setup
    mol, mf = setup_molecule(xyz)
    total_ncas = sum(ncas_list)
    total_nelecas = sum(nelecas_list)
    
    print(f"Molecule: {name}")
    print(f"Fragments: {len(ncas_list)} fragments")
    print(f"Active space: {ncas_list} orbitals, {nelecas_list} electrons per fragment")
    print(f"Total active space: {total_ncas} orbitals, {total_nelecas} electrons")
    print(f"HF energy: {mf.e_tot:.10f} hartree")
    
    results = {}
    
    # 1. CASCI calculation
    print(f"\n1. CASCI({total_ncas},{total_nelecas}) Calculation:")
    try:
        casci_energy, casci_conv = run_casci(mol, mf, total_ncas, total_nelecas)
        results['CASCI'] = casci_energy
        print(f"   Energy: {casci_energy:.10f} hartree")
        print(f"   Converged: {casci_conv}")
        print(f"   Correlation energy: {casci_energy - mf.e_tot:.10f} hartree")
    except Exception as e:
        print(f"   CASCI failed: {e}")
        results['CASCI'] = None
    
    # 2. Single-state LASSI (ground state only)
    print(f"\n2. Single-State LASSI Calculation:")
    try:
        lassi_gs, lasscf_e, lasscf_conv = run_lassi_single_state(
            mol, mf, ncas_list, nelecas_list, spinsub_list, frag_atom_list)
        results['LASSI_single'] = lassi_gs
        print(f"   LASSCF energy: {lasscf_e:.10f} hartree") 
        print(f"   LASSI ground state: {lassi_gs:.10f} hartree")
        print(f"   LASSCF converged: {lasscf_conv}")
        print(f"   Correlation energy: {lassi_gs - mf.e_tot:.10f} hartree")
    except Exception as e:
        print(f"   Single-state LASSI failed: {e}")
        results['LASSI_single'] = None
    
    # 3. Multi-state LASSI 
    print(f"\n3. Multi-State LASSI Calculation:")
    try:
        lassi_gs_multi, all_roots, nstates, lasscf_conv = run_lassi_multi_state(
            mol, mf, ncas_list, nelecas_list, spinsub_list, frag_atom_list)
        results['LASSI_multi'] = lassi_gs_multi
        results['LASSI_excited'] = all_roots[1:] if len(all_roots) > 1 else []
        
        print(f"   Number of states: {nstates}")
        print(f"   LASSI ground state: {lassi_gs_multi:.10f} hartree")
        print(f"   Correlation energy: {lassi_gs_multi - mf.e_tot:.10f} hartree")
        
        if len(all_roots) > 1:
            print(f"   Excitation energies (eV):")
            for i, e in enumerate(all_roots[1:5]):  # Show first 4 excited states
                exc_ev = (e - all_roots[0]) * 27.2114
                print(f"     State {i+1}: {exc_ev:.6f} eV")
                
    except Exception as e:
        print(f"   Multi-state LASSI failed: {e}")
        results['LASSI_multi'] = None
    
    return results

def print_final_comparison(h4_results, h6_results):
    """Print final comparison table"""
    
    print(f"\n{'='*90}")
    print("FINAL GROUND STATE ENERGY COMPARISON")
    print(f"{'='*90}")
    
    methods = [('CASCI', 'CASCI'), 
               ('LASSI (single)', 'LASSI_single'), 
               ('LASSI (multi)', 'LASSI_multi')]
    
    print(f"{'Method':<20} {'H4 Energy (hartree)':<25} {'H6 Energy (hartree)':<25} {'Δ(H6-H4)':<15}")
    print(f"{'-'*20} {'-'*25} {'-'*25} {'-'*15}")
    
    for method_name, key in methods:
        h4_e = h4_results.get(key)
        h6_e = h6_results.get(key)
        
        h4_str = f"{h4_e:.10f}" if h4_e is not None else "Failed"
        h6_str = f"{h6_e:.10f}" if h6_e is not None else "Failed"
        
        if h4_e is not None and h6_e is not None:
            delta_str = f"{h6_e - h4_e:.8f}"
        else:
            delta_str = "N/A"
        
        print(f"{method_name:<20} {h4_str:<25} {h6_str:<25} {delta_str:<15}")
    
    # Energy differences analysis
    print(f"\n{'='*90}")
    print("ENERGY DIFFERENCES ANALYSIS")
    print(f"{'='*90}")
    
    if all(h4_results.get(k) is not None for k in ['CASCI', 'LASSI_single', 'LASSI_multi']):
        print(f"\nH4 Energy Differences (hartree):")
        casci_h4 = h4_results['CASCI']
        lassi_s_h4 = h4_results['LASSI_single']  
        lassi_m_h4 = h4_results['LASSI_multi']
        
        print(f"  LASSI(single) - CASCI:    {lassi_s_h4 - casci_h4:.10f}")
        print(f"  LASSI(multi) - CASCI:     {lassi_m_h4 - casci_h4:.10f}")
        print(f"  LASSI(multi) - LASSI(s):  {lassi_m_h4 - lassi_s_h4:.10f}")
        
        # Energy differences in milli-hartree for better readability
        print(f"\nH4 Energy Differences (milli-hartree):")
        print(f"  LASSI(single) - CASCI:    {(lassi_s_h4 - casci_h4)*1000:.6f}")
        print(f"  LASSI(multi) - CASCI:     {(lassi_m_h4 - casci_h4)*1000:.6f}")
        print(f"  LASSI(multi) - LASSI(s):  {(lassi_m_h4 - lassi_s_h4)*1000:.6f}")
    
    if all(h6_results.get(k) is not None for k in ['CASCI', 'LASSI_single', 'LASSI_multi']):
        print(f"\nH6 Energy Differences (hartree):")
        casci_h6 = h6_results['CASCI']
        lassi_s_h6 = h6_results['LASSI_single']
        lassi_m_h6 = h6_results['LASSI_multi']
        
        print(f"  LASSI(single) - CASCI:    {lassi_s_h6 - casci_h6:.10f}")
        print(f"  LASSI(multi) - CASCI:     {lassi_m_h6 - casci_h6:.10f}")
        print(f"  LASSI(multi) - LASSI(s):  {lassi_m_h6 - lassi_s_h6:.10f}")
        
        # Energy differences in milli-hartree
        print(f"\nH6 Energy Differences (milli-hartree):")
        print(f"  LASSI(single) - CASCI:    {(lassi_s_h6 - casci_h6)*1000:.6f}")
        print(f"  LASSI(multi) - CASCI:     {(lassi_m_h6 - casci_h6)*1000:.6f}")
        print(f"  LASSI(multi) - LASSI(s):  {(lassi_m_h6 - lassi_s_h6)*1000:.6f}")

def main():
    """Main comparison function"""
    print("GROUND STATE ENERGY COMPARISON: CASCI vs LASSI")
    print("Methods compared:")
    print("- CASCI: Complete Active Space Configuration Interaction")
    print("- LASSI: Localized Active Space State Interaction") 
    print("  * Single-state: Ground state only")
    print("  * Multi-state: Ground + excited states with state averaging")
    
    # H4 molecule (2 fragments)
    h4_results = compare_molecule(
        "H4", H4xyz, 
        ncas_list=[2, 2], 
        nelecas_list=[2, 2],
        spinsub_list=[1, 1],
        frag_atom_list=((0, 1), (2, 3))
    )
    
    # H6 molecule (3 fragments) 
    h6_results = compare_molecule(
        "H6", H6xyz,
        ncas_list=[2, 2, 2],
        nelecas_list=[2, 2, 2], 
        spinsub_list=[1, 1, 1],
        frag_atom_list=((0, 1), (2, 3), (4, 5))
    )
    
    # Final comparison
    print_final_comparison(h4_results, h6_results)
    
    return h4_results, h6_results

if __name__ == "__main__":
    results = main()