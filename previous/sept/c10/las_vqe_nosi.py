#!/usr/bin/env python3
"""
LAS-VQE-NOSI: Localized Active Space - Variational Quantum Eigensolver - Non-Orthogonal State Interaction

This class implements a ground state quantum chemistry computation framework that:
1. Performs LASSCF (Localized Active Space Self-Consistent Field)
2. Prepares LAS-VQE states with customizable excitation parameters
3. Performs VQE optimization on different sets of excitation parameters
4. Performs state interaction on LAS-VQE states to obtain the final ground state energy
"""

import numpy as np
from scipy import linalg
from scipy.linalg import eigh
from typing import List, Tuple, Optional, Dict, Any, Union
import time

# PySCF and MRH imports
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.tools import molden
from mrh.exploratory.citools import grad, lasci_ominus1, fockspace
from mrh.exploratory.unitary_cc import lasuccsd

from pyscf.fci import cistring
from pyscf import fci
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci.spin_op import contract_ss


class LASVQENOSIState:
    """Container for individual VQE-optimized states"""
    def __init__(self, psi, energy: float, excitations: Tuple[List, List], converged: bool = False):
        self.psi = psi  # VQE wavefunction
        self.energy = energy  # Individual VQE energy
        self.excitations = excitations  # (a_idxs, i_idxs) excitation parameters
        self.converged = converged  # Convergence status


class LASVQENOSIResult:
    """Container for complete LAS-VQE-NOSI calculation results"""
    def __init__(self):
        # Energy values
        self.rhf_energy: Optional[float] = None
        self.lasscf_energy: Optional[float] = None
        self.casci_energy: Optional[float] = None
        self.individual_vqe_energies: List[float] = []
        self.final_energies: Optional[np.ndarray] = None
        self.ground_state_energy: Optional[float] = None
        
        # State information
        self.n_states: int = 0
        self.ground_state_coefficients: Optional[np.ndarray] = None
        self.hamiltonian_matrix: Optional[np.ndarray] = None
        self.overlap_matrix: Optional[np.ndarray] = None
        
        # Convergence and timing
        self.lasscf_converged: bool = False
        self.vqe_converged: List[bool] = []
        self.total_time: float = 0.0
        
        # Method details
        self.excitation_sets: List[Tuple[List, List]] = []
        self.gradient_threshold: float = 0.0
        self.vqe_max_cycles: int = 0


class LASVQENOSI:
    """
    LAS-VQE-NOSI: Ground State Computation with Non-Orthogonal State Interaction
    
    This class performs ground state computation for molecules using:
    1. LASSCF as reference method
    2. Customizable LAS-VQE state preparation with excitation parameter sets
    3. VQE optimization for different excitation parameter sets
    4. State interaction between VQE-optimized states for final energy
    """
    
    def __init__(self, mol: Union[gto.Mole, callable], ncas_sub: Tuple, nelec_sub: Tuple,
                 basis: str = '6-31g', frag_atom_list: Optional[Tuple] = None,
                 gradient_threshold: float = 1e-4, vqe_max_cycles: int = 10,
                 verbose: int = 1):
        """
        Initialize LAS-VQE-NOSI calculation
        
        Parameters:
        -----------
        mol : pyscf.Mole or callable
            Molecule object or function that creates molecule
        ncas_sub : tuple
            Active space sizes for each fragment, e.g., (3, 3) for two fragments with 3 orbitals each
        nelec_sub : tuple  
            Number of electrons for each fragment, e.g., ((2,1),(1,2)) for alpha,beta electrons
        basis : str
            Basis set name (default: '6-31g')
        frag_atom_list : tuple, optional
            Fragment atom indices for localization, e.g., ((0,1,2), (3,4,5))
        gradient_threshold : float
            Threshold for excitation selection (default: 1e-4)
        vqe_max_cycles : int
            Maximum VQE optimization cycles (default: 10)  
        verbose : int
            Verbosity level (default: lib.logger.INFO)
        """
        self.mol = mol
        self.ncas_sub = ncas_sub
        self.nelec_sub = nelec_sub
        self.basis = basis
        self.frag_atom_list = frag_atom_list
        self.gradient_threshold = gradient_threshold
        self.vqe_max_cycles = vqe_max_cycles
        self.verbose = verbose
        
        # Calculation objects
        self.mf = None  # Mean-field object
        self.las = None  # LASSCF object
        self.casci = None  # CASCI reference
        
        # VQE state containers
        self.vqe_states: List[LASVQENOSIState] = []
        
        # Excitation parameter sets - users can directly assign to this
        self.excitation_parameter_sets: List[Tuple[List, List]] = []
        
        # Results
        self.result = LASVQENOSIResult()
        
    def setup_molecule(self) -> gto.Mole:
        """Setup molecule and perform mean-field calculation"""
        # Handle callable vs. pre-built molecule
        if callable(self.mol) and not isinstance(self.mol, gto.Mole):
            self.mol = self.mol(0, 0, self.basis)
        
        # Ensure molecule is built
        if not hasattr(self.mol, '_built') or not self.mol._built:
            self.mol.verbose = self.verbose
            self.mol.build()
        
        # Mean-field calculation
        self.mf = scf.RHF(self.mol).run()
        self.result.rhf_energy = self.mf.e_tot
        
        if self.verbose >= 1:
            print(f"RHF energy: {self.mf.e_tot:.10f} hartree")
            
        return self.mol
    
    def perform_lasscf(self, mo_guess: Optional[np.ndarray] = None) -> LASSCF:
        """
        Perform LASSCF calculation
        
        Parameters:
        -----------
        mo_guess : np.ndarray, optional
            Initial molecular orbital guess
            
        Returns:
        --------
        las : LASSCF
            Converged LASSCF object
        """
        if self.mf is None:
            raise ValueError("Must setup molecule first. Call setup_molecule().")
            
        self.las = LASSCF(self.mf, self.ncas_sub, self.nelec_sub)
        
        # Initialize molecular orbitals
        if mo_guess is not None:
            mo = mo_guess
        elif self.frag_atom_list is not None:
            mo = self.las.localize_init_guess(self.frag_atom_list, self.mf.mo_coeff)
        else:
            # Use default sorting - user should provide frag_atom_list for specific molecules
            print("WARNING: No fragment atom list provided, using default MO ordering for LASSCF guess.")
            mo = self.mf.mo_coeff
            
        # Run LASSCF
        self.las.kernel(mo)
        self.result.lasscf_energy = self.las.e_tot
        self.result.lasscf_converged = self.las.converged
        
        if self.verbose >= 1:
            print(f"LASSCF energy: {self.las.e_tot:.10f} hartree")
            if not self.las.converged:
                print("WARNING: LASSCF did not converge!")
                
        return self.las
    
    def setup_casci_reference(self) -> mcscf.CASCI:
        """Setup CASCI reference for comparison"""
        if self.las is None:
            raise ValueError("Must perform LASSCF first. Call perform_lasscf().")
            
        ncas_total = sum(self.ncas_sub)
        nelec_total = sum([sum(ne) for ne in self.nelec_sub])
        
        self.casci = mcscf.CASCI(self.mf, ncas_total, nelec_total).set(
            fcisolver=csf_solver(self.mol, smult=1))
        self.casci.kernel(self.las.mo_coeff)
        self.result.casci_energy = self.casci.e_tot
        
        if self.verbose >= 1:
            print(f"CASCI({ncas_total},{nelec_total}) energy: {self.casci.e_tot:.10f} hartree")
            
        return self.casci
    
    def get_default_excitation_sets(self) -> List[Tuple[List, List]]:
        """
        Generate default excitation parameter sets using gradient-based selection
        This is a helper method - users can call this to get default sets, then modify as needed
        
        Returns:
        --------
        excitation_sets : List[Tuple[List, List]]
            Default excitation parameter sets based on gradient selection
        """
        if self.las is None:
            raise ValueError("Must perform LASSCF first. Call perform_lasscf().")
            
        if self.verbose >= 1:
            print(f"Generating default excitations using gradient threshold: {self.gradient_threshold}")
            
        # Compute gradients for excitation selection
        tmplas = LASSCF(self.mf, self.ncas_sub, self.las.nelecas_sub)
        tmplas.mo_coeff = self.las.mo_coeff
        tmplas.ci = self.las.ci
        
        all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(
            tmplas, self.gradient_threshold)
            
        if self.verbose >= 1:
            print(f"Selected {len(a_idxs_selected)} total excitations")
        
        # Split excitations into two sets (simple strategy)
        n_total = len(a_idxs_selected)
        if n_total < 2:
            raise ValueError(f"Only {n_total} excitations selected. Need at least 2 for state interaction.")
            
        # Set 1: First half of excitations
        split_point = n_total // 2
        a_idxs_set1 = a_idxs_selected[:split_point] if split_point > 0 else a_idxs_selected[:1]
        i_idxs_set1 = i_idxs_selected[:split_point] if split_point > 0 else i_idxs_selected[:1]
        
        # Set 2: Second half of excitations  
        a_idxs_set2 = a_idxs_selected[split_point:] if n_total > split_point else a_idxs_selected[1:]
        i_idxs_set2 = i_idxs_selected[split_point:] if n_total > split_point else i_idxs_selected[1:]
        
        # Ensure minimum excitations per set
        min_excitations = 1
        if len(a_idxs_set1) < min_excitations:
            a_idxs_set1, i_idxs_set1 = a_idxs_selected[:min_excitations], i_idxs_selected[:min_excitations]
        if len(a_idxs_set2) < min_excitations:
            a_idxs_set2, i_idxs_set2 = a_idxs_selected[-min_excitations:], i_idxs_selected[-min_excitations:]
            
        default_sets = [
            (a_idxs_set1, i_idxs_set1),
            (a_idxs_set2, i_idxs_set2)
        ]
        
        if self.verbose >= 1:
            print(f"Generated 2 default excitation parameter sets:")
            print(f"  Set 1: {len(a_idxs_set1)} excitations")
            print(f"  Set 2: {len(a_idxs_set2)} excitations")
            
        return default_sets
    
    def perform_vqe_on_sets(self) -> List[LASVQENOSIState]:
        """
        Perform VQE optimization on different sets of excitation parameters
        
        Note: Users must assign excitation parameter sets to self.excitation_parameter_sets before calling this method
        
        Returns:
        --------
        vqe_states : List[LASVQENOSIState]
            List of VQE-optimized states
        """
        if not self.excitation_parameter_sets:
            raise ValueError("Must assign excitation parameter sets to self.excitation_parameter_sets first.")
            
        if self.verbose >= 1:
            print(f"Performing VQE optimization on {len(self.excitation_parameter_sets)} parameter sets...")
            print(f"VQE settings: max_cycles={self.vqe_max_cycles}")
            
        self.vqe_states = []
        lasci_ominus1.GLOBAL_MAX_CYCLE = self.vqe_max_cycles
        
        ncas = sum(self.las.ncas_sub)
        nelecas_sub = [sum(nelec) for nelec in self.las.nelecas_sub]
        nelecas_total = sum(nelecas_sub)
        
        for i, (a_idxs, i_idxs) in enumerate(self.excitation_parameter_sets):
            if self.verbose >= 1:
                print(f"\n--- VQE Optimization Set {i+1} ---")
                print(f"Excitations: {len(a_idxs)} doubles, {len(i_idxs)} singles")
            
            # Setup VQE solver for this parameter set
            mc_uscc = mcscf.CASCI(self.mf, ncas, nelecas_total)
            mc_uscc.mo_coeff = self.las.mo_coeff
            mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(self.mol, a_idxs, i_idxs)
            mc_uscc.fcisolver.norb_f = self.las.ncas_sub
            
            # Prepare initial CI guess
            ci0 = self._cilas2f(self.las.ci, self.las.ncas_sub, self.las.nelecas_sub)
            
            # Run VQE optimization
            try:
                e_vqe = mc_uscc.kernel(ci0=ci0)[0]
                psi = mc_uscc.fcisolver.psi
                converged = getattr(mc_uscc.fcisolver, 'converged', False)
                
                vqe_state = LASVQENOSIState(psi, e_vqe, (a_idxs, i_idxs), converged)
                self.vqe_states.append(vqe_state)
                self.result.individual_vqe_energies.append(e_vqe)
                self.result.vqe_converged.append(converged)
                
                if self.verbose >= 1:
                    print(f"Set {i+1} VQE energy: {e_vqe:.10f} hartree")
                    if not converged:
                        print(f"  WARNING: Set {i+1} VQE did not converge")
                        
            except Exception as e:
                if self.verbose >= 1:
                    print(f"ERROR in VQE set {i+1}: {str(e)}")
                # Create dummy state for failed optimization
                dummy_state = LASVQENOSIState(None, float('inf'), (a_idxs, i_idxs), False)
                self.vqe_states.append(dummy_state)
                self.result.individual_vqe_energies.append(float('inf'))
                self.result.vqe_converged.append(False)
        
        # Store the last mc_uscc for matrix element calculations
        self._mc_uscc = mc_uscc
        self.result.n_states = len(self.vqe_states)
        self.result.excitation_sets = self.excitation_parameter_sets
        
        if self.verbose >= 1:
            print(f"VQE optimization completed for {len(self.vqe_states)} states")
            print(f"Individual VQE energies: {self.result.individual_vqe_energies}")
            
        return self.vqe_states
    
    def perform_state_interaction(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform state interaction on LAS-VQE states to get final energy
        
        Returns:
        --------
        final_energies : np.ndarray
            Final energies from state interaction
        eigenvectors : np.ndarray  
            Eigenvectors of the state interaction
        """
        if not self.vqe_states or not any(state.psi is not None for state in self.vqe_states):
            raise ValueError("Must perform VQE on parameter sets first. Call perform_vqe_on_sets().")
            
        if self.verbose >= 1:
            print("\nPerforming state interaction...")
            
        # Filter out failed states
        valid_states = [state for state in self.vqe_states if state.psi is not None]
        if len(valid_states) < 2:
            raise ValueError(f"Need at least 2 valid VQE states for interaction, got {len(valid_states)}")
            
        # Get effective Hamiltonian
        h1eff, e_core = self._mc_uscc.get_h1eff(self._mc_uscc.mo_coeff)
        h2eff = self._mc_uscc.get_h2eff()
        h = [e_core, h1eff, h2eff]
        
        # Build Hamiltonian and overlap matrices
        nc = len(valid_states)
        S_matrix = np.zeros((nc, nc), dtype=np.complex128)
        H_matrix = np.zeros((nc, nc), dtype=np.complex128)
        
        for i in range(nc):
            for j in range(nc):
                S_matrix[i, j], H_matrix[i, j] = self._get_Sij_Hij(
                    valid_states[i].psi, valid_states[j].psi, h)
        
        if self.verbose >= 2:
            print("Overlap matrix S:")
            self._print_complex_matrix(S_matrix)
            print("Hamiltonian matrix H:")
            self._print_complex_matrix(H_matrix)
            
        # Store matrices
        self.result.overlap_matrix = S_matrix
        self.result.hamiltonian_matrix = H_matrix
        
        # Solve generalized eigenvalue problem H|c> = E S|c>
        final_energies, eigenvectors = eigh(H_matrix, S_matrix)
        
        self.result.final_energies = final_energies
        self.result.ground_state_energy = final_energies[0]
        self.result.ground_state_coefficients = eigenvectors[:, 0]
        
        if self.verbose >= 1:
            print(f"Final energies from state interaction: {final_energies}")
            print(f"Ground state energy: {final_energies[0]:.10f} hartree")
            print(f"Ground state coefficients: {eigenvectors[:, 0]}")
            
        return final_energies, eigenvectors
    
    def run_full_calculation(self, mo_guess: Optional[np.ndarray] = None, 
                           use_default_excitations: bool = True) -> LASVQENOSIResult:
        """
        Run the complete LAS-VQE-NOSI calculation
        
        Parameters:
        -----------
        mo_guess : np.ndarray, optional
            Initial molecular orbital guess for LASSCF
        use_default_excitations : bool, optional
            If True, automatically generates default excitation sets (if none assigned)
            If False, user must assign to self.excitation_parameter_sets before calling
            
        Returns:
        --------
        result : LASVQENOSIResult
            Complete calculation results
        """
        start_time = time.time()
        
        if self.verbose >= 1:
            print("=== Starting LAS-VQE-NOSI Calculation ===")
            
        try:
            # Step 1: Setup molecule and mean-field
            if self.verbose >= 1:
                print("\n1. Setting up molecule and mean-field...")
            self.setup_molecule()
            
            # Step 2: Perform LASSCF
            if self.verbose >= 1:
                print("\n2. Performing LASSCF...")
            self.perform_lasscf(mo_guess)
            
            # Step 3: Setup CASCI reference
            if self.verbose >= 1:
                print("\n3. Setting up CASCI reference...")
            self.setup_casci_reference()
            
            # Step 4: Check/generate excitation parameter sets
            if not self.excitation_parameter_sets and use_default_excitations:
                if self.verbose >= 1:
                    print("\n4. No excitation sets assigned, generating defaults...")
                self.excitation_parameter_sets = self.get_default_excitation_sets()
            elif not self.excitation_parameter_sets:
                raise ValueError("No excitation parameter sets assigned. Set self.excitation_parameter_sets or use use_default_excitations=True")
            elif self.verbose >= 1:
                print(f"\n4. Using {len(self.excitation_parameter_sets)} user-assigned excitation parameter sets")
                for i, (a_idxs, i_idxs) in enumerate(self.excitation_parameter_sets):
                    print(f"   Set {i}: {len(a_idxs)} excitations")
            
            # Step 5: Perform VQE on parameter sets
            if self.verbose >= 1:
                print("\n5. Performing VQE optimization on parameter sets...")
            self.perform_vqe_on_sets()
            
            # Step 6: Perform state interaction
            if self.verbose >= 1:
                print("\n6. Performing state interaction...")
            self.perform_state_interaction()
            
            if self.verbose >= 1:
                print("\n=== LAS-VQE-NOSI Calculation Complete ===")
                self._print_summary()
            
        except Exception as e:
            if self.verbose >= 1:
                print(f"ERROR in LAS-VQE-NOSI calculation: {str(e)}")
            raise
            
        finally:
            self.result.total_time = time.time() - start_time
            self.result.gradient_threshold = self.gradient_threshold
            self.result.vqe_max_cycles = self.vqe_max_cycles
            
        return self.result
    
    def _print_summary(self):
        """Print calculation summary"""
        print(f"\nLAS-VQE-NOSI Results Summary:")
        print(f"  RHF energy:              {self.result.rhf_energy:.10f} hartree")
        print(f"  LASSCF energy:           {self.result.lasscf_energy:.10f} hartree") 
        print(f"  CASCI energy:            {self.result.casci_energy:.10f} hartree")
        print(f"  Ground state energy:     {self.result.ground_state_energy:.10f} hartree")
        if len(self.result.individual_vqe_energies) > 0:
            print(f"  Individual VQE energies: {[f'{e:.10f}' for e in self.result.individual_vqe_energies]}")
        
        # Energy differences
        if self.result.casci_energy and self.result.ground_state_energy:
            diff_casci = (self.result.ground_state_energy - self.result.casci_energy) * 1000
            print(f"  Correlation vs CASCI:    {diff_casci:.3f} mEh")
        if self.result.lasscf_energy and self.result.ground_state_energy:
            diff_lasscf = (self.result.ground_state_energy - self.result.lasscf_energy) * 1000  
            print(f"  Correlation vs LASSCF:   {diff_lasscf:.3f} mEh")
            
        print(f"  Total time:              {self.result.total_time:.2f} seconds")
    
    # Helper methods
    def _cilas2f(self, lasci, norb_f, nelec_f):
        """Convert LAS CI to Fock space representation"""
        ci_f = []
        for i, ci in enumerate(lasci):
            ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
        return ci_f

    def _get_Sij_Hij(self, psi_i, psi_j, h):
        """Compute overlap and Hamiltonian matrix elements between VQE states"""
        ucj, hucj = psi_j.hc_x(psi_j.x, h)[1:3]
        uci, huci = psi_i.hc_x(psi_i.x, h)[1:3]
        ucj, hucj = ucj.ravel(), hucj.ravel()
        uci = uci.ravel()
        Sij = uci.conj().dot(ucj)
        Hij = uci.conj().dot(hucj)
        return Sij, Hij

    def _print_complex_matrix(self, mat):
        """Print complex matrix in a readable format"""
        for row in mat:
            print("  ".join(f"{x.real:+.8f}{x.imag:+.8f}j" 
                           if x.imag != 0 else f"{x.real:+.8f}+{x.imag:.8f}j" 
                           for x in row))


# Example usage and testing
def test_h4_example():
    """Test LAS-VQE-NOSI on H4 molecule"""
    
    # H4 geometry
    h4_xyz = """H      0.000000000000   0.000000000000   0.000000000000
H      1.000000000000   0.000000000000   0.000000000000
H      0.273746762116   2.195450598147   0.100000000000
H      1.232912762116   1.895450598147  -0.100000000000"""
    
    # Create molecule
    mol = gto.M(atom=h4_xyz, basis='sto-3g', verbose=0)
    
    # LAS-VQE-NOSI calculation
    calc = LASVQENOSI(
        mol=mol,
        ncas_sub=(2, 2),  # 2 orbitals per fragment
        nelec_sub=((1, 1), (1, 1)),  # 1 alpha, 1 beta electron per fragment
        basis='sto-3g',
        frag_atom_list=((0, 1), (2, 3)),  # Fragment atom indices
        gradient_threshold=1e-4,
        vqe_max_cycles=5,
        verbose=1  # Use integer instead of lib.logger.INFO
    )
    
    # Run calculation
    result = calc.run_full_calculation()
    
    return result


if __name__ == "__main__":
    print("Testing LAS-VQE-NOSI on H4 molecule...")
    result = test_h4_example()
    
    print(f"\nTest completed successfully!")
    print(f"Final ground state energy: {result.ground_state_energy:.10f} hartree")