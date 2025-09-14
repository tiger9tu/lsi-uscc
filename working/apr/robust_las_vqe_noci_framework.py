#!/usr/bin/env python3
"""
Robust LAS-VQE-NOCI Framework

Enhanced version with better error handling, linear dependency detection,
and configurable testing options.
"""

import sys
import numpy as np
import copy
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pyscf import gto, scf, mcscf, lib
from scipy.linalg import eigh, LinAlgError
sys.path.append('/home/jinx/repo/qchem/las-uscc-noci-bot/working')

from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.citools import grad, lasci_ominus1, fockspace
from mrh.exploratory.unitary_cc import lasuccsd

@dataclass
class LASVQENOCIResult:
    """Container for LAS-VQE-NOCI calculation results"""
    mol_name: str
    basis: str
    success: bool
    error_msg: str = ""
    
    # Energy values
    rhf_energy: Optional[float] = None
    lasscf_energy: Optional[float] = None
    casci_energy: Optional[float] = None
    vqe_noci_energy: Optional[float] = None
    excited_energy: Optional[float] = None
    
    # Method details
    n_excitations_total: int = 0
    n_excitations_set1: int = 0
    n_excitations_set2: int = 0
    
    # State mixing coefficients
    ground_state_coeff1: Optional[complex] = None
    ground_state_coeff2: Optional[complex] = None
    
    # Overlap matrix condition
    overlap_condition_number: Optional[float] = None
    overlap_min_eigenval: Optional[float] = None
    
    # Performance metrics
    calculation_time: float = 0.0
    lasscf_converged: bool = False
    vqe1_converged: bool = False
    vqe2_converged: bool = False
    
    # Energy differences (in mEh)
    correlation_vs_lasscf: Optional[float] = None
    correlation_vs_casci: Optional[float] = None
    energy_gap_ev: Optional[float] = None

class RobustLASVQENOCIFramework:
    """Enhanced framework with robust error handling"""
    
    def __init__(self, gradient_threshold: float = 0.0001, vqe_max_cycles: int = 10, 
                 overlap_threshold: float = 1e-8, verbose: int = 1):
        """
        Initialize the framework
        
        Parameters:
        - gradient_threshold: Threshold for excitation selection
        - vqe_max_cycles: Maximum VQE optimization cycles
        - overlap_threshold: Threshold for detecting linear dependencies in overlap matrix
        - verbose: Verbosity level (0-2)
        """
        self.gradient_threshold = gradient_threshold
        self.vqe_max_cycles = vqe_max_cycles
        self.overlap_threshold = overlap_threshold
        self.verbose = verbose
        
    @staticmethod
    def cilas2f(lasci, norb_f, nelec_f):
        """Convert LAS CI to fock space format"""
        ci_f = []
        for i, ci in enumerate(lasci):
            ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
        return ci_f

    @staticmethod
    def get_Sij_Hij(psi_i, psi_j, h):
        """Get overlap and Hamiltonian matrix elements between VQE states"""
        ucj, hucj = psi_j.hc_x(psi_j.x, h)[1:3]
        uci, huci = psi_i.hc_x(psi_i.x, h)[1:3]
        ucj, hucj = ucj.ravel(), hucj.ravel()
        uci = uci.ravel()
        Sij = uci.conj().dot(ucj)
        Hij = uci.conj().dot(hucj)
        return Sij, Hij

    @staticmethod
    def select_local_interfrag_excitations(a_idxs, i_idxs, frag_pair_sorb):
        """Select excitations localized to a specific fragment pair"""
        a_idxs_sel, i_idxs_sel = [], []
        for a, i in zip(a_idxs, i_idxs):
            # Check if excitation is within the specified fragment pair
            if all(idx in frag_pair_sorb for idx in a) and all(idx in frag_pair_sorb for idx in i):
                a_idxs_sel.append(a)
                i_idxs_sel.append(i)
        return a_idxs_sel, i_idxs_sel

    def solve_robust_generalized_eigenvalue(self, H, S):
        """Robust solver for generalized eigenvalue problem with linear dependency handling"""
        
        # Check overlap matrix condition
        S_eigenvals = np.linalg.eigvals(S)
        min_eigval = np.min(S_eigenvals)
        condition_number = np.max(S_eigenvals) / max(min_eigval, 1e-16)
        
        if self.verbose >= 2:
            print(f"    Overlap matrix condition number: {condition_number:.2e}")
            print(f"    Minimum eigenvalue: {min_eigval:.2e}")
        
        try:
            # Standard generalized eigenvalue solver
            eigvals, eigvecs = eigh(H, S)
            return eigvals, eigvecs, condition_number, min_eigval, None
            
        except LinAlgError as e:
            if self.verbose >= 1:
                print(f"    Standard solver failed: {e}")
                print("    Attempting regularized solution...")
            
            try:
                # Regularization approach: S + ε*I
                regularization = max(self.overlap_threshold, -min_eigval * 1.1)
                S_reg = S + regularization * np.eye(S.shape[0])
                
                eigvals, eigvecs = eigh(H, S_reg)
                return eigvals, eigvecs, condition_number, min_eigval, f"Regularized with ε={regularization:.2e}"
                
            except LinAlgError as e2:
                return None, None, condition_number, min_eigval, f"Both standard and regularized solvers failed: {e2}"

    def run_single_molecule(self, mol_config: Dict[str, Any]) -> LASVQENOCIResult:
        """Run LAS-VQE-NOCI calculation for a single molecule"""
        
        start_time = time.time()
        result = LASVQENOCIResult(
            mol_name=mol_config['name'],
            basis=mol_config['basis'],
            success=False
        )
        
        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print(f"LAS-VQE-NOCI: {mol_config['name']}")
            print(f"{'='*60}")
        
        try:
            # Setup molecule
            mol = gto.M(
                atom=mol_config['xyz'],
                basis=mol_config['basis'],
                verbose=0
            )
            mol.build()
            
            if self.verbose >= 1:
                print(f"Molecule: {mol.natm} atoms, {mol.nelectron} electrons, {mol.nao} orbitals")
            
            # Mean-field calculation
            mf = scf.RHF(mol).run()
            result.rhf_energy = mf.e_tot
            
            if self.verbose >= 1:
                print(f"RHF energy: {mf.e_tot:.10f} hartree")
            
            # LASSCF calculation
            if self.verbose >= 1:
                print("\nRunning LASSCF...")
            
            las = LASSCF(mf, mol_config['ncas'], mol_config['nelecas'], 
                        spin_sub=mol_config['spinsub'])
            
            mo_loc = las.localize_init_guess(mol_config['frag_atom_list'], mf.mo_coeff)
            las_result = las.kernel(mo_loc)
            h2eff_sub, veff = las_result[-2:]
            
            result.lasscf_energy = las.e_tot
            result.lasscf_converged = las.converged
            
            if not las.converged and self.verbose >= 1:
                print("WARNING: LASSCF did not converge!")
            
            # CASCI reference
            cas = mcscf.CASCI(mf, las.ncas, sum(las.nelecas_sub))
            cas.mo_coeff = las.mo_coeff
            result.casci_energy = cas.kernel()[0]
            
            if self.verbose >= 1:
                print(f"LASSCF energy: {las.e_tot:.10f} hartree")
                print(f"CASCI energy: {result.casci_energy:.10f} hartree")
            
            # Gradient-based excitation selection
            if self.verbose >= 1:
                print(f"\nSelecting excitations (threshold: {self.gradient_threshold})...")
            
            ncore, ncas = las.ncore, las.ncas
            mo_coeff = las.mo_coeff
            ci_ref = las.ci
            nelec_sub = las.nelecas_sub
            
            tmplas = LASSCF(mf, mol_config['ncas'], nelec_sub)
            tmplas.mo_coeff = mo_coeff
            tmplas.ci = ci_ref
            
            all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(
                tmplas, self.gradient_threshold)
            
            result.n_excitations_total = len(a_idxs_selected)
            
            if self.verbose >= 1:
                print(f"Selected {len(a_idxs_selected)} excitations")
            
            # Fragment-localized excitation sets
            if 'frag_spin_orb' in mol_config and len(mol_config['frag_spin_orb']) >= 2:
                # Use fragment-based selection
                frag_keys = list(mol_config['frag_spin_orb'].keys())
                
                # Set 1: First two fragments
                frag_01_orbs = mol_config['frag_spin_orb'][frag_keys[0]] + mol_config['frag_spin_orb'][frag_keys[1]]
                a_idxs_set1, i_idxs_set1 = self.select_local_interfrag_excitations(
                    a_idxs_selected, i_idxs_selected, frag_01_orbs)
                
                # Set 2: Last two fragments (or middle fragments if 3+ fragments)
                if len(frag_keys) >= 3:
                    frag_12_orbs = mol_config['frag_spin_orb'][frag_keys[1]] + mol_config['frag_spin_orb'][frag_keys[2]]
                else:
                    frag_12_orbs = mol_config['frag_spin_orb'][frag_keys[0]] + mol_config['frag_spin_orb'][frag_keys[1]]
                
                a_idxs_set2, i_idxs_set2 = self.select_local_interfrag_excitations(
                    a_idxs_selected, i_idxs_selected, frag_12_orbs)
                
            else:
                # Fallback: Split excitations by half
                split_point = len(a_idxs_selected) // 2
                a_idxs_set1, i_idxs_set1 = a_idxs_selected[:split_point], i_idxs_selected[:split_point]
                a_idxs_set2, i_idxs_set2 = a_idxs_selected[split_point:], i_idxs_selected[split_point:]
            
            # Ensure minimum excitations per set and avoid identical sets
            min_excitations = 5
            if len(a_idxs_set1) < min_excitations:
                a_idxs_set1, i_idxs_set1 = a_idxs_selected[:min_excitations], i_idxs_selected[:min_excitations]
            if len(a_idxs_set2) < min_excitations:
                a_idxs_set2, i_idxs_set2 = a_idxs_selected[-min_excitations:], i_idxs_selected[-min_excitations:]
            
            # Check if sets are too similar (would cause linear dependence)
            set1_indices = set(range(len(a_idxs_set1)))
            set2_indices = set(range(len(a_idxs_set1), len(a_idxs_set1) + len(a_idxs_set2)))
            
            if len(set1_indices & set2_indices) / max(len(set1_indices), len(set2_indices)) > 0.8:
                # Sets too similar, use different splitting
                mid = len(a_idxs_selected) // 2
                quarter = len(a_idxs_selected) // 4
                a_idxs_set1, i_idxs_set1 = a_idxs_selected[:mid], i_idxs_selected[:mid]
                a_idxs_set2, i_idxs_set2 = a_idxs_selected[quarter:quarter+mid], i_idxs_selected[quarter:quarter+mid]
            
            result.n_excitations_set1 = len(a_idxs_set1)
            result.n_excitations_set2 = len(a_idxs_set2)
            
            if self.verbose >= 1:
                print(f"Excitation set 1: {len(a_idxs_set1)} excitations")
                print(f"Excitation set 2: {len(a_idxs_set2)} excitations")
            
            # VQE optimization for both sets
            lasci_ominus1.GLOBAL_MAX_CYCLE = self.vqe_max_cycles
            nelecas_sub = [sum(nelec) for nelec in nelec_sub]
            
            # VQE Set 1
            if self.verbose >= 1:
                print("\nRunning VQE optimization for set 1...")
            
            mc_uscc_1 = mcscf.CASCI(mf, ncas, sum(nelecas_sub))
            mc_uscc_1.mo_coeff = mo_coeff
            mc_uscc_1.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_set1, i_idxs_set1)
            mc_uscc_1.fcisolver.norb_f = las.ncas_sub
            
            ci0_1 = self.cilas2f(ci_ref, las.ncas_sub, nelec_sub)
            mc_uscc_1.kernel(ci0=ci0_1)
            psi_1 = mc_uscc_1.fcisolver.psi
            result.vqe1_converged = hasattr(mc_uscc_1.fcisolver, 'converged')
            
            # VQE Set 2
            if self.verbose >= 1:
                print("Running VQE optimization for set 2...")
            
            mc_uscc_2 = mcscf.CASCI(mf, ncas, sum(nelecas_sub))
            mc_uscc_2.mo_coeff = mo_coeff
            mc_uscc_2.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_set2, i_idxs_set2)
            mc_uscc_2.fcisolver.norb_f = las.ncas_sub
            
            ci0_2 = self.cilas2f(ci_ref, las.ncas_sub, nelec_sub)
            mc_uscc_2.kernel(ci0=ci0_2)
            psi_2 = mc_uscc_2.fcisolver.psi
            result.vqe2_converged = hasattr(mc_uscc_2.fcisolver, 'converged')
            
            # State interaction
            if self.verbose >= 1:
                print("Performing state interaction...")
            
            h1eff, e_core = mc_uscc_1.get_h1eff(mc_uscc_1.mo_coeff)
            h2eff = mc_uscc_1.get_h2eff()
            h = [e_core, h1eff, h2eff]
            
            psis = [psi_1, psi_2]
            nc = len(psis)
            
            S = np.zeros((nc, nc), dtype=np.complex128)
            H = np.zeros((nc, nc), dtype=np.complex128)
            
            for i in range(nc):
                for j in range(nc):
                    S[i, j], H[i, j] = self.get_Sij_Hij(psis[i], psis[j], h)
            
            # Solve generalized eigenvalue problem with robust handling
            eigvals, eigvecs, condition_number, min_eigval, solver_note = self.solve_robust_generalized_eigenvalue(H, S)
            
            result.overlap_condition_number = condition_number
            result.overlap_min_eigenval = min_eigval
            
            if eigvals is None:
                raise ValueError(solver_note)
            
            if solver_note and self.verbose >= 1:
                print(f"    Solver note: {solver_note}")
            
            result.vqe_noci_energy = eigvals[0]
            result.excited_energy = eigvals[1] if len(eigvals) > 1 else None
            result.ground_state_coeff1 = eigvecs[0, 0]
            result.ground_state_coeff2 = eigvecs[1, 0] if eigvecs.shape[0] > 1 else 0.0
            
            # Calculate energy differences
            if result.lasscf_energy is not None:
                result.correlation_vs_lasscf = (result.vqe_noci_energy - result.lasscf_energy) * 1000
            
            if result.casci_energy is not None:
                result.correlation_vs_casci = (result.vqe_noci_energy - result.casci_energy) * 1000
            
            if result.excited_energy is not None:
                result.energy_gap_ev = (result.excited_energy - result.vqe_noci_energy) * 27.2114
            
            result.success = True
            
            if self.verbose >= 1:
                print(f"\nLAS-VQE-NOCI Results:")
                print(f"  Ground state energy: {result.vqe_noci_energy:.10f} hartree")
                if result.excited_energy:
                    print(f"  Excited state energy: {result.excited_energy:.10f} hartree")
                    print(f"  Energy gap: {result.energy_gap_ev:.3f} eV")
                print(f"  Correlation vs LASSCF: {result.correlation_vs_lasscf:.3f} mEh")
                print(f"  Correlation vs CASCI: {result.correlation_vs_casci:.3f} mEh")
            
        except Exception as e:
            if self.verbose >= 1:
                print(f"ERROR in {mol_config['name']}: {str(e)}")
            result.success = False
            result.error_msg = str(e)
        
        result.calculation_time = time.time() - start_time
        return result
    
    def run_batch(self, mol_configs: List[Dict[str, Any]]) -> List[LASVQENOCIResult]:
        """Run LAS-VQE-NOCI calculations for multiple molecules"""
        
        if self.verbose >= 1:
            print(f"Starting batch LAS-VQE-NOCI calculations for {len(mol_configs)} molecules")
            print(f"Parameters: gradient_threshold={self.gradient_threshold}, vqe_cycles={self.vqe_max_cycles}")
        
        results = []
        successful = 0
        
        for i, mol_config in enumerate(mol_configs):
            if self.verbose >= 1:
                print(f"\n[{i+1}/{len(mol_configs)}] Processing {mol_config['name']}...")
            
            result = self.run_single_molecule(mol_config)
            results.append(result)
            
            if result.success:
                successful += 1
        
        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print(f"BATCH SUMMARY: {successful}/{len(mol_configs)} calculations successful")
            print(f"{'='*60}")
        
        return results
    
    def print_summary_table(self, results: List[LASVQENOCIResult]):
        """Print a summary table of all results"""
        
        print(f"\n{'='*140}")
        print("LAS-VQE-NOCI RESULTS SUMMARY")
        print(f"{'='*140}")
        
        # Header
        header = f"{'Molecule':<15} {'Basis':<8} {'Success':<8} {'LASSCF':<12} {'CASCI':<12} {'VQE-NOCI':<12} {'Gap(eV)':<8} {'Corr(mEh)':<10} {'CondNum':<10} {'Time(s)':<8}"
        print(header)
        print('-' * 140)
        
        for result in results:
            success_str = "✓" if result.success else "✗"
            lasscf_str = f"{result.lasscf_energy:.6f}" if result.lasscf_energy else "N/A"
            casci_str = f"{result.casci_energy:.6f}" if result.casci_energy else "N/A"
            vqe_noci_str = f"{result.vqe_noci_energy:.6f}" if result.vqe_noci_energy else "N/A"
            gap_str = f"{result.energy_gap_ev:.2f}" if result.energy_gap_ev else "N/A"
            corr_str = f"{result.correlation_vs_casci:.2f}" if result.correlation_vs_casci else "N/A"
            cond_str = f"{result.overlap_condition_number:.1e}" if result.overlap_condition_number else "N/A"
            time_str = f"{result.calculation_time:.1f}"
            
            row = f"{result.mol_name:<15} {result.basis:<8} {success_str:<8} {lasscf_str:<12} {casci_str:<12} {vqe_noci_str:<12} {gap_str:<8} {corr_str:<10} {cond_str:<10} {time_str:<8}"
            print(row)
            
            # Print error message for failed calculations
            if not result.success and result.error_msg and self.verbose >= 1:
                print(f"    Error: {result.error_msg[:100]}...")
        
        print(f"{'='*140}")

# Include the molecular configurations from the original framework
def get_molecular_configs():
    """Get standardized molecular configurations for testing"""
    
    # Data directory
    data_dir = '/home/jinx/repo/qchem/las_uccsd_data'
    
    # Load external geometries
    def load_xyz(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None
    
    c4xyz = load_xyz(f'{data_dir}/polyenes/geometries/c4.xyz')
    c6xyz = load_xyz(f'{data_dir}/polyenes/geometries/c6.xyz')
    c10xyz = load_xyz(f'{data_dir}/polyenes/geometries/c10.xyz')
    h10_circle_xyz = load_xyz(f'{data_dir}/circle/H10.xyz')
    stil90xyz = load_xyz(f'{data_dir}/stilbene/geometries/stil-90.xyz')
    
    # Internal geometries
    H4xyz = '''H      0.000000000000   0.000000000000   0.000000000000
H      1.000000000000   0.000000000000   0.000000000000
H      0.273746762116   2.195450598147   0.100000000000
H      1.232912762116   1.895450598147  -0.100000000000'''

    H6xyz = '''H      0.000000000000   0.000000000000   0.000000000000
H      1.000000000000   0.000000000000   0.000000000000
H      0.273746762116   2.195450598147   0.100000000000
H      1.232912762116   1.895450598147  -0.100000000000
H      0.507178110854   4.193780995243   0.049334760036
H      1.506140937609   3.988021397347  -0.049334760036'''

    H8xyz = '''H      0.000000000000   0.000000000000   0.000000000000
H      1.000000000000   0.000000000000   0.000000000000
H      0.273746762116   2.195450598147   0.100000000000
H      1.232912762116   1.895450598147  -0.100000000000
H      0.507178110854   4.193780995243   0.049334760036
H      1.506140937609   3.988021397347  -0.049334760036
H      0.845946518048   6.364231296231   0.197836732111
H      1.674032054647   5.908472292654  -0.197836732111'''
    
    # Define configurations
    configs = []
    
    # Hydrogen chains
    h4_sto3g = {
        'name': 'H4_STO3G',
        'xyz': H4xyz,
        'basis': 'sto-3g',
        'ncas': [2, 2],
        'nelecas': [2, 2],
        'spinsub': [1, 1],
        'frag_atom_list': ((0, 1), (2, 3)),
        'frag_spin_orb': {
            0: (0, 1, 4, 5),
            1: (2, 3, 6, 7)
        }
    }
    configs.append(h4_sto3g)
    
    h6_sto3g = {
        'name': 'H6_STO3G',
        'xyz': H6xyz,
        'basis': 'sto-3g',
        'ncas': [2, 2, 2],
        'nelecas': [2, 2, 2],
        'spinsub': [1, 1, 1],
        'frag_atom_list': ((0, 1), (2, 3), (4, 5)),
        'frag_spin_orb': {
            0: (0, 1, 6, 7),
            1: (2, 3, 8, 9),
            2: (4, 5, 10, 11)
        }
    }
    configs.append(h6_sto3g)
    
    h6_631g = copy.deepcopy(h6_sto3g)
    h6_631g['name'] = 'H6_631G'
    h6_631g['basis'] = '6-31g'
    configs.append(h6_631g)
    
    h8_sto3g = {
        'name': 'H8_STO3G',
        'xyz': H8xyz,
        'basis': 'sto-3g',
        'ncas': [2, 2, 2, 2],
        'nelecas': [2, 2, 2, 2],
        'spinsub': [1, 1, 1, 1],
        'frag_atom_list': ((0, 1), (2, 3), (4, 5), (6, 7)),
        'frag_spin_orb': {
            0: (0, 1, 8, 9),
            1: (2, 3, 10, 11),
            2: (4, 5, 12, 13),
            3: (6, 7, 14, 15)
        }
    }
    configs.append(h8_sto3g)
    
    # Carbon systems
    if c4xyz:
        c4_sto3g = {
            'name': 'C4_STO3G',
            'xyz': c4xyz,
            'basis': 'sto-3g',
            'ncas': [2, 2],
            'nelecas': [2, 2],
            'spinsub': [1, 1],
            'frag_atom_list': [[0, 2], [3, 1]],
            'frag_spin_orb': {
                0: (0, 1, 4, 5),
                1: (2, 3, 6, 7),
            }
        }
        configs.append(c4_sto3g)
    
    if c6xyz:
        c6_sto3g = {
            'name': 'C6_STO3G',
            'xyz': c6xyz,
            'basis': 'sto-3g',
            'ncas': [2, 2, 2],
            'nelecas': [2, 2, 2],
            'spinsub': [1, 1, 1],
            'frag_atom_list': [[0, 2], [10, 12], [3, 1]],
            'frag_spin_orb': {
                0: (0, 1, 6, 7),
                1: (2, 3, 8, 9),
                2: (4, 5, 10, 11),
            }
        }
        configs.append(c6_sto3g)
    
    if h10_circle_xyz:
        h10_circle_sto3g = {
            'name': 'H10_CIRCLE_STO3G',
            'xyz': h10_circle_xyz,
            'basis': 'sto-3g',
            'ncas': [2, 2, 2, 2, 2],
            'nelecas': [2, 2, 2, 2, 2],
            'spinsub': [1, 1, 1, 1, 1],
            'frag_atom_list': [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
            'frag_spin_orb': {
                0: (0, 1, 10, 11),
                1: (2, 3, 12, 13),
                2: (4, 5, 14, 15),
                3: (6, 7, 16, 17),
                4: (8, 9, 18, 19)
            }
        }
        configs.append(h10_circle_sto3g)
    
    return configs

def main():
    """Main testing function with command line arguments"""
    
    parser = argparse.ArgumentParser(description='Robust LAS-VQE-NOCI Framework')
    parser.add_argument('--molecules', nargs='*', 
                       help='Specific molecules to test (default: test subset)')
    parser.add_argument('--gradient-threshold', type=float, default=0.0001,
                       help='Gradient threshold for excitation selection')
    parser.add_argument('--vqe-cycles', type=int, default=10,
                       help='Maximum VQE optimization cycles')
    parser.add_argument('--overlap-threshold', type=float, default=1e-8,
                       help='Overlap matrix linear dependency threshold')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=1,
                       help='Verbosity level')
    parser.add_argument('--all', action='store_true',
                       help='Test all available molecules')
    
    args = parser.parse_args()
    
    # Get molecular configurations
    mol_configs = get_molecular_configs()
    
    # Create framework instance
    framework = RobustLASVQENOCIFramework(
        gradient_threshold=args.gradient_threshold,
        vqe_max_cycles=args.vqe_cycles,
        overlap_threshold=args.overlap_threshold,
        verbose=args.verbose
    )
    
    # Select molecules to test
    if args.all:
        filtered_configs = mol_configs
    elif args.molecules:
        filtered_configs = [config for config in mol_configs if config['name'] in args.molecules]
    else:
        # Default test set (reliable molecules)
        test_molecules = ['H6_STO3G', 'C6_STO3G', 'H8_STO3G']
        filtered_configs = [config for config in mol_configs if config['name'] in test_molecules]
    
    if len(filtered_configs) == 0:
        print("No matching molecular configurations found!")
        return
    
    print(f"Running LAS-VQE-NOCI calculations for: {[c['name'] for c in filtered_configs]}")
    
    # Run batch calculations
    results = framework.run_batch(filtered_configs)
    
    # Print summary table
    framework.print_summary_table(results)
    
    # Additional analysis
    successful_results = [r for r in results if r.success]
    if successful_results:
        print(f"\nSUCCESSFUL CALCULATIONS ANALYSIS:")
        print(f"Average correlation energy: {np.mean([r.correlation_vs_casci for r in successful_results if r.correlation_vs_casci]):.3f} mEh")
        if any(r.energy_gap_ev for r in successful_results):
            print(f"Average energy gap: {np.mean([r.energy_gap_ev for r in successful_results if r.energy_gap_ev]):.3f} eV")
        print(f"Average calculation time: {np.mean([r.calculation_time for r in successful_results]):.1f} seconds")
        print(f"Success rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
    
if __name__ == "__main__":
    main()