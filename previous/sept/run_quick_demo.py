#!/usr/bin/env python3
"""
Quick demonstration script to run basic calculations on all molecules
This will run just CASCI and LASSCF for each system to verify functionality
"""

import os
import sys
import time
from pyscf import gto, scf, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

def load_geometry(geom_file):
    """Load geometry from file"""
    with open(geom_file, 'r') as f:
        return f.read()

def run_casci_lasscf(name, geom_file, ncas_sub, nelec_sub, spin_sub, frag_atom_list):
    """Run CASCI and LASSCF for a molecule"""
    print(f"\n{'='*60}")
    print(f"Running {name} calculations")
    print(f"{'='*60}")
    
    try:
        # Setup molecule
        atom_string = load_geometry(geom_file)
        mol = gto.M(atom=atom_string, basis='6-31g', verbose=0)
        mol.build()
        
        print(f"Molecule: {name}")
        print(f"Atoms: {mol.natm}, Electrons: {mol.nelectron}, Basis functions: {mol.nao}")
        
        # Mean-field calculation
        mf = scf.RHF(mol)
        mf.kernel()
        
        results = {}
        
        # CASCI calculation
        print(f"\nRunning CASCI...")
        start_time = time.time()
        ncas_total = sum(ncas_sub)
        nelec_total = sum(sum(pair) for pair in nelec_sub)
        mc = mcscf.CASCI(mf, ncas_total, nelec_total)
        mc.kernel()
        casci_time = time.time() - start_time
        
        results['CASCI'] = {
            'energy': mc.e_tot,
            'time': casci_time,
            'converged': mc.converged
        }
        
        print(f"CASCI: {mc.e_tot:.10f} hartree ({casci_time:.2f}s, {'✓' if mc.converged else '✗'})")
        
        # LASSCF calculation
        print(f"Running LASSCF...")
        start_time = time.time()
        las = LASSCF(mf, ncas_sub, nelec_sub, spin_sub=spin_sub)
        las.state_average_(weights=[1.0], charges=[0]*len(ncas_sub), 
                         spins=[0]*len(ncas_sub), smults=[1]*len(ncas_sub))
        
        # Use localized orbitals
        from pyscf.lo import orth
        mo_coeff_loc = orth.orth_ao(mol, method='meta_lowdin')
        las.kernel(mo_coeff_loc)
        lasscf_time = time.time() - start_time
        
        results['LASSCF'] = {
            'energy': las.e_tot,
            'time': lasscf_time,
            'converged': las.converged
        }
        
        print(f"LASSCF: {las.e_tot:.10f} hartree ({lasscf_time:.2f}s, {'✓' if las.converged else '✗'})")
        
        if results['CASCI']['converged'] and results['LASSCF']['converged']:
            delta = (results['LASSCF']['energy'] - results['CASCI']['energy']) * 1000
            print(f"Energy difference (LASSCF - CASCI): {delta:+.3f} mEh")
        
        return results
        
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        return None

def main():
    """Run quick demonstration on all molecules"""
    
    molecules = {
        'C8': {
            'geom_file': '/home/jinx/repo/qchem/las_uccsd_data/polyenes/geometries/c8.xyz',
            'ncas_sub': (2, 2, 2, 2),
            'nelec_sub': ((1, 1), (1, 1), (1, 1), (1, 1)),
            'spin_sub': (1, 1, 1, 1),
            'frag_atom_list': [[0,2], [10,12], [13,11], [3,1]]
        },
        'C10': {
            'geom_file': '/home/jinx/repo/qchem/las_uccsd_data/polyenes/geometries/c10.xyz',
            'ncas_sub': (2, 2, 2, 2, 2),
            'nelec_sub': ((1, 1), (1, 1), (1, 1), (1, 1), (1, 1)),
            'spin_sub': (1, 1, 1, 1, 1),
            'frag_atom_list': [[0,2], [10,12], [18,19], [13,11], [3,1]]
        },
        'Stilbene-001': {
            'geom_file': '/home/jinx/repo/qchem/las_uccsd_data/stilbene/geometries/stil-001.xyz',
            'ncas_sub': (4, 2, 4),
            'nelec_sub': ((2, 2), (1, 1), (2, 2)),
            'spin_sub': (1, 1, 1),
            'frag_atom_list': [[1,2,3,4,5,6,15,16,17,18,19], [0,7,14,20], [8,9,10,11,12,13,21,22,23,24,25]]
        }
    }
    
    print("🚀 Quick LAS calculations demonstration")
    print("=" * 60)
    
    all_results = {}
    
    for name, config in molecules.items():
        result = run_casci_lasscf(
            name=name,
            geom_file=config['geom_file'],
            ncas_sub=config['ncas_sub'],
            nelec_sub=config['nelec_sub'],
            spin_sub=config['spin_sub'],
            frag_atom_list=config['frag_atom_list']
        )
        all_results[name] = result
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY RESULTS")
    print(f"{'='*60}")
    
    print("| Molecule | CASCI Energy | LASSCF Energy | ΔE (mEh) | Status |")
    print("|----------|--------------|---------------|----------|--------|")
    
    for name, result in all_results.items():
        if result and result['CASCI']['converged'] and result['LASSCF']['converged']:
            casci_e = result['CASCI']['energy']
            lasscf_e = result['LASSCF']['energy']
            delta = (lasscf_e - casci_e) * 1000
            status = "✅ Success"
            print(f"| {name:8s} | {casci_e:.8f} | {lasscf_e:.8f} | {delta:+6.3f} | {status} |")
        else:
            print(f"| {name:8s} | {'N/A':>12s} | {'N/A':>13s} | {'N/A':>8s} | ❌ Failed |")
    
    print(f"\n✅ Quick demonstration completed!")

if __name__ == "__main__":
    main()