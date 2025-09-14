#!/usr/bin/env python3
"""
Test additional systems: C6 6-31G, Stilbene 6-31G, and C2H4N4 with VQE max cycles = 5
"""

import numpy as np
from scipy.linalg import eigh
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSIrq

from mrh.exploratory.citools import grad, lasci_ominus1, fockspace
from mrh.exploratory.unitary_cc import lasuccsd

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

def calculate_all_methods(name, xyz, ncas_sub, nelecas_sub, spinsub, frag_atom_list, basis='6-31g', vqe_cycles=5):
    """Calculate all four methods for one molecule"""
    print(f"\n{'='*80}")
    print(f"COMPLETE ANALYSIS: {name}")
    print(f"{'='*80}")
    
    # Setup molecule
    mol = gto.M(atom=xyz, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    
    ncas_total = sum(ncas_sub)
    nelec_total = sum(nelecas_sub)
    
    print(f"System: {ncas_total} orbitals, {nelec_total} electrons in {len(ncas_sub)} fragments")
    print(f"Fragment atom list: {frag_atom_list}")
    print(f"RHF energy: {mf.e_tot:.10f} hartree")
    
    results = {}
    
    # 1. LASSCF
    print("1. LASSCF...")
    las = LASSCF(mf, ncas_sub, nelecas_sub, spin_sub=spinsub)
    mo_loc = las.localize_init_guess(frag_atom_list, mf.mo_coeff)
    las.kernel(mo_loc)
    results['lasscf'] = las.e_tot
    print(f"   Energy: {las.e_tot:.10f} hartree")
    
    # 2. CASCI
    print("2. CASCI...")
    mc_casci = mcscf.CASCI(mf, ncas_total, nelec_total).set(
        fcisolver=csf_solver(mol, smult=1))
    mc_casci.kernel(las.mo_coeff)
    results['casci'] = mc_casci.e_tot
    print(f"   Energy: {mc_casci.e_tot:.10f} hartree")
    
    # 3. LASSIrq
    print("3. LASSIrq[1,1]...")
    lsi = LASSIrq(las, r=1, q=1)
    e_roots, si_rq = lsi.kernel()
    results['lassirq'] = e_roots[0]
    num_states = len(e_roots)
    print(f"   Energy: {e_roots[0]:.10f} hartree ({num_states} states)")
    
    # 4. LASSIrq-VQE
    print(f"4. LASSIrq-VQE (max_cycle={vqe_cycles})...")
    try:
        # Get electron configurations
        nelec_fr = lsi.get_nelec_frs()
        ncore, ncas = las.ncore, las.ncas
        nelecas = sum(las.nelecas_sub)
        mo_coeff = las.mo_coeff
        
        # VQE optimization for each CT state
        nstates = len(lsi.ci[0])
        print(f"   Number of CT states: {nstates}")
        psis = []
        vqe_individual_energies = []  # Store individual VQE energies
        
        for i in range(nstates):
            ci = [[lsi.ci[fragj][i]] for fragj in range(len(lsi.ci))]
            nelec_sub = [nelec_fr[fragj][i] for fragj in range(len(nelec_fr))]
            
            # Temporary LASSCF for gradient
            tmplas = LASSCF(mf, ncas_sub, nelec_sub)
            tmplas.mo_coeff = mo_coeff
            tmplas.ci = ci
            all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(tmplas, 0.00001)
            
            # VQE solver setup
            nelecas_sub = [sum(nelec) for nelec in nelec_sub]
            mc_uscc = mcscf.CASCI(mf, ncas, nelecas)
            mc_uscc.mo_coeff = mo_coeff
            mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
            mc_uscc.fcisolver.norb_f = las.ncas_sub
            lasci_ominus1.GLOBAL_MAX_CYCLE = vqe_cycles
            mc_uscc.kernel(ci0=cilas2f(ci, las.ncas_sub, nelec_sub))
            psis.append(mc_uscc.fcisolver.psi)
            vqe_individual_energies.append(mc_uscc.e_tot)  # Store individual VQE energy
        
        # Matrix elements between VQE states
        h1eff, e_core = mc_uscc.get_h1eff(mc_uscc.mo_coeff)
        h2eff = mc_uscc.get_h2eff()
        h = [e_core, h1eff, h2eff]
        
        nc = len(psis)
        S = np.zeros((nc, nc), dtype=np.complex128)
        H = np.zeros((nc, nc), dtype=np.complex128)
        
        for i in range(nc):
            for j in range(nc):
                S[i, j], H[i, j] = get_Sij_Hij(psis[i], psis[j], h)
        
        # Find lowest individual VQE energy before state interaction
        lowest_individual_vqe = min(vqe_individual_energies)
        print(f"   Lowest individual VQE energy: {lowest_individual_vqe:.10f} hartree")
        
        # Solve eigenvalue problem
        eigvals, eigvecs = eigh(H, S)
        results['lassirq_vqe'] = eigvals[0]
        results['lowest_individual_vqe'] = lowest_individual_vqe
        print(f"   Final LASSIrq-VQE energy (after state interaction): {eigvals[0]:.10f} hartree")
        
        vqe_si_improvement = eigvals[0] - lowest_individual_vqe
        print(f"   State interaction improvement: {vqe_si_improvement:+.10f} hartree")
        
    except Exception as e:
        print(f"   LASSIrq-VQE failed: {e}")
        results['lassirq_vqe'] = None
        results['lowest_individual_vqe'] = None
    
    # Summary
    print(f"\n{'='*60}")
    print(f"ENERGY SUMMARY FOR {name}")
    print(f"{'='*60}")
    
    print(f"{'Method':<15} {'Energy (hartree)':<20} {'Rel to LASSCF (mH)':<20}")
    print("-" * 60)
    
    lasscf_energy = results['lasscf']
    methods = [('casci', 'CASCI'), ('lassirq_vqe', 'LASSIrq-VQE'), 
               ('lassirq', 'LASSIrq'), ('lasscf', 'LASSCF')]
    
    for method, label in methods:
        if results.get(method) is not None:
            energy = results[method]
            rel_energy = (energy - lasscf_energy) * 1000  # Convert to milli-hartree
            print(f"{label:<15} {energy:<20.10f} {rel_energy:+18.6f}")
    
    if results.get('lowest_individual_vqe'):
        energy = results['lowest_individual_vqe']
        rel_energy = (energy - lasscf_energy) * 1000
        print(f"{'VQE (indiv)':<15} {energy:<20.10f} {rel_energy:+18.6f}")
    
    # Energy ordering
    energies = []
    labels = []
    for method, label in methods:
        if results.get(method) is not None:
            energies.append(results[method])
            labels.append(label)
    
    if len(energies) >= 3:
        sorted_pairs = sorted(zip(energies, labels))
        print(f"\nEnergy ordering (lowest to highest):")
        for i, (energy, label) in enumerate(sorted_pairs):
            print(f"  {i+1}. {label:<15}: {energy:.10f} hartree")
        
        # Check correct ordering
        casci_ok = results.get('casci') is not None
        vqe_ok = results.get('lassirq_vqe') is not None
        lassirq_ok = results.get('lassirq') is not None
        
        if casci_ok and vqe_ok and lassirq_ok:
            correct = (results['casci'] <= results['lassirq_vqe'] <= 
                      results['lassirq'] <= results['lasscf'])
            print(f"\nCorrect ordering (CASCI ≤ LASSIrq-VQE ≤ LASSIrq ≤ LASSCF): {'✓' if correct else '✗'}")
    
    return results

def main():
    """Test all additional molecular systems"""
    
    print("=" * 100)
    print("ADDITIONAL SYSTEMS ENERGY COMPARISON")
    print("C6 6-31G, Stilbene 6-31G, C2H4N4 with VQE max_cycle=5")
    print("=" * 100)
    
    # Read geometries
    data_dir = '/home/jinx/repo/qchem/las_uccsd_data'
    
    # C6 geometry
    try:
        with open(f'{data_dir}/polyenes/geometries/c6.xyz', 'r') as f:
            c6_xyz = f.read()
    except FileNotFoundError:
        print("C6 geometry file not found, skipping...")
        c6_xyz = None
    
    # Stilbene geometry  
    try:
        with open(f'{data_dir}/stilbene/geometries/stil-90.xyz', 'r') as f:
            stil_xyz = f.read()
    except FileNotFoundError:
        print("Stilbene geometry file not found, skipping...")
        stil_xyz = None
    
    # C2H4N4 geometry (using local structure)
    from c2h4n4_struct import structure as c2h4n4_struct
    
    all_results = []
    
    # 1. C6 6-31G
    if c6_xyz:
        try:
            results = calculate_all_methods(
                name='C6_631G',
                xyz=c6_xyz,
                ncas_sub=(2, 2, 2),
                nelecas_sub=(2, 2, 2),
                spinsub=[1, 1, 1],
                frag_atom_list=((0, 1), (2, 3), (4, 5)),  # Three C=C fragments
                basis='6-31g',
                vqe_cycles=5
            )
            all_results.append(results)
        except Exception as e:
            print(f"C6 6-31G calculation failed: {e}")
    
    # 2. Stilbene 6-31G (90 degree rotation)
    if stil_xyz:
        try:
            results = calculate_all_methods(
                name='STIL_631G_90',
                xyz=stil_xyz,
                ncas_sub=(4, 2, 4),
                nelecas_sub=(4, 2, 4), 
                spinsub=[1, 1, 1],
                frag_atom_list=((0, 1, 2, 3, 4, 5), (6, 7), (8, 9, 10, 11, 12, 13)),  # Phenyl-vinyl-phenyl
                basis='6-31g',
                vqe_cycles=5
            )
            all_results.append(results)
        except Exception as e:
            print(f"Stilbene 6-31G calculation failed: {e}")
    
    # 3. C2H4N4 6-31G
    try:
        mol_c2h4n4 = c2h4n4_struct(0, 0, '6-31g')
        results = calculate_all_methods(
            name='C2H4N4_631G',
            xyz=mol_c2h4n4.atom,
            ncas_sub=(3, 3),
            nelecas_sub=((2, 1), (1, 2)),
            spinsub=[1, 1],
            frag_atom_list=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)),  # Two fragments
            basis='6-31g',
            vqe_cycles=5
        )
        all_results.append(results)
    except Exception as e:
        print(f"C2H4N4 6-31G calculation failed: {e}")
    
    # Final summary
    print(f"\n{'='*100}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*100}")
    
    if all_results:
        print(f"{'System':<15} {'LASSCF':<12} {'LASSIrq':<12} {'VQE(indiv)':<12} {'LASSIrq-VQE':<12} {'CASCI':<12}")
        print("-" * 80)
        
        for r in all_results:
            name = r.get('name', 'Unknown')
            print(f"{name:<15} ", end="")
            print(f"{r['lasscf']:.6f} " if r.get('lasscf') else "FAILED    ", end="")
            print(f"{r['lassirq']:.6f} " if r.get('lassirq') else "FAILED    ", end="") 
            print(f"{r['lowest_individual_vqe']:.6f} " if r.get('lowest_individual_vqe') else "FAILED    ", end="")
            print(f"{r['lassirq_vqe']:.6f} " if r.get('lassirq_vqe') else "FAILED    ", end="")
            print(f"{r['casci']:.6f}" if r.get('casci') else "FAILED    ")
    
    return all_results

if __name__ == "__main__":
    results = main()