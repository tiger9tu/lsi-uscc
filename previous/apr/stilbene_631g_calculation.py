#!/usr/bin/env python3
"""
Stilbene 6-31G LASSIRQ-VQE Energy Comparison
Run comprehensive energy comparison for stilbene with 6-31G basis
"""

import numpy as np
import time
from scipy.linalg import eigh
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSIrq

from mrh.exploratory.citools import grad, lasci_ominus1, fockspace
from mrh.exploratory.unitary_cc import lasuccsd

def read_xyz_file(filepath):
    """Read XYZ file content"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None

def setup_molecule(config):
    """Set up molecule from configuration"""
    mol = gto.Mole()
    mol.atom = config['xyz']
    mol.basis = config['basis']
    mol.verbose = 4
    mol.build()
    
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.kernel()
    
    return mol, mf

def cilas2f(lasci, norb_f, nelec_f):
    """Convert LAS CI to Fock space representation"""
    ci_f = []
    for i, ci in enumerate(lasci):
        ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
    return ci_f

def get_Sij_Hij(psi_i, psi_j, h):
    """Compute overlap and Hamiltonian matrix elements"""
    ucj, hucj = psi_j.hc_x(psi_j.x, h)[1:3]
    uci, huci = psi_i.hc_x(psi_i.x, h)[1:3]
    ucj, hucj = ucj.ravel(), hucj.ravel()
    uci = uci.ravel()
    Sij = uci.conj().dot(ucj)
    Hij = uci.conj().dot(hucj)
    return Sij, Hij

def calculate_lassirq_vqe_energy(las, lsi, mol, mf, max_cycle=3):
    """Calculate LASSIrq-VQE energy using VQE optimization"""
    try:
        # Get necessary parameters
        ncore, ncas = las.ncore, las.ncas
        nelecas = sum(las.nelecas_sub)
        mo_coeff = las.mo_coeff
        nelec_fr = lsi.get_nelec_frs()
        
        # VQE optimization for each CT state
        nstates = len(lsi.ci[0])
        print(f"Running VQE optimization for {nstates} charge transfer states")
        
        vqe_energies_individual = []
        all_psi_vqe = []
        
        for istate in range(nstates):
            print(f"\n=== VQE State {istate} ===")
            
            # Set up CI solver for this state
            ci_vec_list = []
            for ifrag in range(len(lsi.ci)):
                ci_vec_list.append(lsi.ci[ifrag][istate])
            
            # Convert to Fock space
            norb_f = [ncas[ifrag] for ifrag in range(len(ncas))]
            nelec_f = [[nelec_fr[ifrag][istate][0], nelec_fr[ifrag][istate][1]] for ifrag in range(len(nelec_fr))]
            ci_f = cilas2f(ci_vec_list, norb_f, nelec_f)
            
            # Set up VQE solver
            ucc = lasuccsd.FCISolver_USCC(nfrag=len(ncas), norb_list=norb_f, nelec_list=nelec_f)
            ucc.conv_tol = 1e-8
            ucc.max_cycle = max_cycle
            ucc.verbose = 4
            ucc.degen_rtol = 1e-10
            
            # Initial guess from LASSIrq CI
            ucc.ci = ci_f
            
            # Run VQE optimization
            h2eff = las.get_h2eff(mo_coeff)
            h1e_las = las.get_h1las()
            energy_vqe = ucc.kernel(h1e_las, h2eff, norb_f, nelec_f, ci0=ci_f)[0]
            
            print(f"State {istate} VQE energy: {energy_vqe:.10f}")
            vqe_energies_individual.append(energy_vqe)
            all_psi_vqe.append(ucc)
        
        print(f"\nIndividual VQE energies before state interaction:")
        for i, e in enumerate(vqe_energies_individual):
            print(f"State {i}: {e:.10f}")
        
        lowest_individual_vqe = min(vqe_energies_individual)
        print(f"Lowest individual VQE energy: {lowest_individual_vqe:.10f}")
        
        # State interaction between VQE-optimized states
        print("\n=== State Interaction ===")
        nstates = len(all_psi_vqe)
        H_matrix = np.zeros((nstates, nstates))
        S_matrix = np.zeros((nstates, nstates))
        
        h2eff = las.get_h2eff(mo_coeff)
        h1e_las = las.get_h1las()
        h = (h1e_las, h2eff)
        
        for i in range(nstates):
            for j in range(nstates):
                if i == j:
                    # Diagonal elements
                    H_matrix[i, j] = vqe_energies_individual[i]
                    S_matrix[i, j] = 1.0
                else:
                    # Off-diagonal elements
                    S_matrix[i, j], H_matrix[i, j] = get_Sij_Hij(all_psi_vqe[i], all_psi_vqe[j], h)
        
        print("H matrix:")
        print(H_matrix)
        print("S matrix:")
        print(S_matrix)
        
        # Solve generalized eigenvalue problem
        eigenvals, eigenvecs = eigh(H_matrix, S_matrix)
        final_energy = eigenvals[0] + las.e_tot - sum(las.e_states)
        
        print(f"State interaction eigenvalues: {eigenvals}")
        print(f"Final LASSIrq-VQE energy: {final_energy:.10f}")
        
        return final_energy, lowest_individual_vqe, vqe_energies_individual
        
    except Exception as e:
        print(f"Error in LASSIrq-VQE calculation: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def run_energy_comparison():
    """Run complete energy comparison for stilbene 6-31G"""
    
    # Load stilbene geometry
    data_dir = '/home/jinx/repo/qchem/las_uccsd_data'
    stil90xyz = read_xyz_file(f'{data_dir}/stilbene/geometries/stil-90.xyz')
    
    if not stil90xyz:
        print("Error: Could not find stilbene geometry file")
        return
    
    # Define stilbene 6-31G configuration
    config = {
        'name': 'STILBENE_631G_90',
        'xyz': stil90xyz,
        'basis': '6-31g',
        'ncas': [4,2,4],
        'nelecas': [4,2,4],
        'spinsub': [1, 1, 1],  # Spin multiplicity = 2S+1 = 1 (singlet)
        'frag_atom_list': [[1,2,3,4,5,6,15,16,17,18,19], [0,7,14,20], [8,9,10,11,12,13,21,22,23,24,25]],
    }
    
    print(f"Running energy comparison for {config['name']}")
    print(f"Active space: {config['ncas']}")
    print(f"Fragment atoms: {config['frag_atom_list']}")
    
    # Setup molecule
    mol, mf = setup_molecule(config)
    print(f"SCF energy: {mf.e_tot:.10f}")
    
    energies = {}
    
    # 1. LASSCF
    print("\n" + "="*50)
    print("1. LASSCF Calculation")
    print("="*50)
    
    las = LASSCF(mf, config['ncas'], config['nelecas'], spin_sub=config['spinsub'])
    frag_atom_list = config['frag_atom_list']
    mo_coeff = las.localize_init_guess(frag_atom_list, mf.mo_coeff)
    las.kernel(mo_coeff)
    
    if not las.converged:
        print("WARNING: LASSCF did not converge!")
    
    energies['LASSCF'] = las.e_tot
    print(f"LASSCF energy: {energies['LASSCF']:.10f}")
    
    # 2. LASSIrq  
    print("\n" + "="*50)
    print("2. LASSIrq Calculation")
    print("="*50)
    
    lsi = LASSIrq(las, r=1, q=1)
    e_roots, si_rq = lsi.kernel()
    energies['LASSIrq'] = e_roots[0]
    print(f"LASSIrq energy: {energies['LASSIrq']:.10f}")
    print(f"Number of LASSIrq states: {len(e_roots)}")
    
    # 3. LASSIrq-VQE
    print("\n" + "="*50)
    print("3. LASSIrq-VQE Calculation")
    print("="*50)
    
    lassirq_vqe_energy, lowest_individual_vqe, individual_vqe_energies = calculate_lassirq_vqe_energy(las, lsi, mol, mf)
    
    if lassirq_vqe_energy is not None:
        energies['LASSIrq-VQE'] = lassirq_vqe_energy
        energies['Lowest_VQE_Individual'] = lowest_individual_vqe
        print(f"LASSIrq-VQE energy: {energies['LASSIrq-VQE']:.10f}")
        print(f"Lowest individual VQE energy: {energies['Lowest_VQE_Individual']:.10f}")
    
    # 4. CASCI
    print("\n" + "="*50)
    print("4. CASCI Calculation")
    print("="*50)
    
    total_ncas = sum(config['ncas'])
    total_nelecas = sum(config['nelecas'])
    
    mc = mcscf.CASCI(mf, total_ncas, total_nelecas)
    mc.fcisolver = csf_solver(mol, smult=1)
    mc.mo_coeff = las.mo_coeff
    mc.kernel()
    energies['CASCI'] = mc.e_tot
    print(f"CASCI energy: {energies['CASCI']:.10f}")
    
    # Print summary
    print("\n" + "="*60)
    print("ENERGY COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Method':<20} {'Energy (a.u.)':<20} {'Rel. Energy (mEh)':<15}")
    print("-"*60)
    
    # Reference energy (CASCI should be lowest)
    ref_energy = energies['CASCI']
    
    methods_order = ['CASCI', 'LASSIrq-VQE', 'Lowest_VQE_Individual', 'LASSIrq', 'LASSCF']
    
    for method in methods_order:
        if method in energies:
            energy = energies[method]
            rel_energy = (energy - ref_energy) * 1000  # Convert to mEh
            print(f"{method:<20} {energy:<20.10f} {rel_energy:<15.3f}")
    
    # Check energy ordering
    print("\n" + "="*60)
    print("ENERGY ORDERING ANALYSIS")
    print("="*60)
    
    expected_order = ['CASCI', 'LASSIrq-VQE', 'LASSIrq', 'LASSCF']
    actual_energies = [energies[method] for method in expected_order if method in energies]
    
    if all(actual_energies[i] <= actual_energies[i+1] for i in range(len(actual_energies)-1)):
        print("✓ Energy ordering is correct: CASCI ≤ LASSIrq-VQE ≤ LASSIrq ≤ LASSCF")
    else:
        print("✗ Energy ordering violation detected!")
        
    print(f"\nCalculation completed for {config['name']}")
    
    # Individual VQE energies analysis
    if individual_vqe_energies:
        print("\n" + "="*60)
        print("INDIVIDUAL VQE ENERGIES ANALYSIS")
        print("="*60)
        print(f"Number of charge transfer states: {len(individual_vqe_energies)}")
        for i, e in enumerate(individual_vqe_energies):
            print(f"State {i}: {e:.10f} a.u.")
        print(f"Lowest: {min(individual_vqe_energies):.10f} a.u.")
        print(f"Highest: {max(individual_vqe_energies):.10f} a.u.")
        print(f"Range: {(max(individual_vqe_energies) - min(individual_vqe_energies))*1000:.3f} mEh")
        
        # State interaction improvement
        if 'LASSIrq-VQE' in energies and 'Lowest_VQE_Individual' in energies:
            improvement = (energies['Lowest_VQE_Individual'] - energies['LASSIrq-VQE']) * 1000
            print(f"State interaction improvement: {improvement:.3f} mEh")

if __name__ == "__main__":
    run_energy_comparison()