#!/usr/bin/env python3
"""
Quick Complete Comparison: Just H4 and H6 with all 4 methods
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

def calculate_all_four_methods(name, xyz, ncas_sub, nelecas_sub, spinsub, frag_atom_list, basis='sto-3g'):
    """Calculate all four methods for one molecule"""
    print(f"\n{'='*60}")
    print(f"COMPLETE ANALYSIS: {name}")
    print(f"{'='*60}")
    
    # Setup molecule
    mol = gto.M(atom=xyz, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    
    ncas_total = sum(ncas_sub)
    nelec_total = sum(nelecas_sub)
    
    print(f"System: {ncas_total} orbitals, {nelec_total} electrons in {len(ncas_sub)} fragments")
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
    
    # 4. LASSIrq-VQE (following mylassi.py exactly)
    print("4. LASSIrq-VQE...")
    try:
        # Get electron configurations
        nelec_fr = lsi.get_nelec_frs()
        ncore, ncas = las.ncore, las.ncas
        nelecas = sum(las.nelecas_sub)
        mo_coeff = las.mo_coeff
        
        # VQE optimization for each CT state
        nstates = len(lsi.ci[0])
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
            lasci_ominus1.GLOBAL_MAX_CYCLE = 2  # Reduced for speed
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
    
    return results

def main():
    """Test on H4 and H6"""
    
    # Test molecules
    molecules = [
        {
            'name': 'H4_STO3G',
            'xyz': '''H 0.0 0.0 0.0
                      H 1.0 0.0 0.0
                      H 0.273746762116 2.195450598147 0.100000000000
                      H 1.232912762116 1.895450598147 -0.100000000000''',
            'ncas_sub': (2, 2),
            'nelecas_sub': (2, 2),
            'spinsub': [1, 1],
            'frag_atom_list': ((0, 1), (2, 3)),
        },
        {
            'name': 'H6_STO3G',
            'xyz': '''H 0.0 0.0 0.0
                      H 1.0 0.0 0.0
                      H 0.273746762116 2.195450598147 0.100000000000
                      H 1.232912762116 1.895450598147 -0.100000000000
                      H 0.507178110854 4.193780995243 0.049334760036
                      H 1.506140937609 3.988021397347 -0.049334760036''',
            'ncas_sub': (2, 2, 2),
            'nelecas_sub': (2, 2, 2),
            'spinsub': [1, 1, 1],
            'frag_atom_list': ((0, 1), (2, 3), (4, 5)),
        }
    ]
    
    all_results = []
    
    print("=" * 80)
    print("COMPLETE 4-METHOD ENERGY COMPARISON")
    print("LASSCF vs LASSIrq vs LASSIrq-VQE vs CASCI")
    print("=" * 80)
    
    for mol in molecules:
        results = calculate_all_four_methods(
            mol['name'], mol['xyz'], mol['ncas_sub'], mol['nelecas_sub'], 
            mol['spinsub'], mol['frag_atom_list']
        )
        results['name'] = mol['name']
        all_results.append(results)
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS TABLE")
    print(f"{'='*80}")
    
    print(f"{'Molecule':<12} {'LASSCF':<12} {'LASSIrq':<12} {'VQE(indiv)':<12} {'LASSIrq-VQE':<12} {'CASCI':<12}")
    print("-" * 77)
    
    for r in all_results:
        print(f"{r['name']:<12} ", end="")
        print(f"{r['lasscf']:.6f} ", end="")
        print(f"{r['lassirq']:.6f} ", end="")
        print(f"{r['lowest_individual_vqe']:.6f}" if r.get('lowest_individual_vqe') else "FAILED    ", end=" ")
        print(f"{r['lassirq_vqe']:.6f}" if r['lassirq_vqe'] else "FAILED    ", end=" ")
        print(f"{r['casci']:.6f}")
    
    # Energy differences for successful cases
    print(f"\n{'='*80}")
    print("ENERGY DIFFERENCES ANALYSIS")
    print(f"{'='*80}")
    
    for r in all_results:
        if r['lassirq_vqe'] is not None:
            print(f"\n{r['name']}:")
            print(f"  LASSIrq - LASSCF:       {r['lassirq'] - r['lasscf']:+.8f} hartree")
            print(f"  VQE(indiv) - LASSIrq:   {r['lowest_individual_vqe'] - r['lassirq']:+.8f} hartree")
            print(f"  LASSIrq-VQE - VQE(indiv): {r['lassirq_vqe'] - r['lowest_individual_vqe']:+.8f} hartree (state interaction)")
            print(f"  LASSIrq-VQE - LASSCF:   {r['lassirq_vqe'] - r['lasscf']:+.8f} hartree")
            print(f"  CASCI - LASSIrq-VQE:    {r['casci'] - r['lassirq_vqe']:+.8f} hartree")
            
            # Energy ordering
            energies = [r['casci'], r['lassirq_vqe'], r['lassirq'], r['lasscf']]
            labels = ['CASCI', 'LASSIrq-VQE', 'LASSIrq', 'LASSCF']
            sorted_pairs = sorted(zip(energies, labels))
            
            print(f"  Energy ordering: ", end="")
            print(" < ".join([f"{label}({energy:.6f})" for energy, label in sorted_pairs]))
            
            # Check if correct ordering
            correct = r['casci'] <= r['lassirq_vqe'] <= r['lassirq'] <= r['lasscf']
            print(f"  Correct ordering: {'✓' if correct else '✗'}")

    return all_results

if __name__ == "__main__":
    results = main()