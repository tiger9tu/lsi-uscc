import numpy as np
from scipy import linalg
from scipy.linalg import eigh
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi
from mrh.my_pyscf.tools import molden
from mrh.exploratory.citools import grad, lasci_ominus1, fockspace
from mrh.exploratory.unitary_cc import lasuccsd

from pyscf.fci import cistring
from pyscf import fci
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci.spin_op import contract_ss


class LASSI_VQE:
    """
    LASSI-VQE Class: Implements Localized Active Space State Interaction with 
    Variational Quantum Eigensolver optimization.
    
    Algorithm:
    1. Perform LAS state averaging on different charge transfer states
    2. Partially optimize each state interaction using VQE
    3. Perform state interaction to get final energies
    """
    
    def __init__(self, mol, ncas_sub, nelec_sub, basis='6-31g', 
                 output_file=None, verbose=lib.logger.INFO):
        """
        Initialize LASSI-VQE calculation
        
        Args:
            mol: PySCF molecule object or structure function
            ncas_sub: tuple of active space sizes for each fragment (e.g., (3,3))
            nelec_sub: tuple of electron numbers for each fragment (e.g., ((2,1),(1,2)))
            basis: basis set name
            output_file: output log file name
            verbose: verbosity level
        """
        self.mol = mol
        self.ncas_sub = ncas_sub
        self.nelec_sub = nelec_sub
        self.basis = basis
        self.output_file = output_file
        self.verbose = verbose
        
        # Initialize calculation objects
        self.mf = None
        self.las = None
        self.las2 = None
        self.mc_casci = None
        self.mc_uscc = None
        self.lsi = None
        
        # Results storage
        self.e_states = None
        self.ham_eff = None
        self.ovlp_eff = None
        self.s2_eff = None
        self.psis = None
        self.H_matrix = None
        self.S_matrix = None
        self.final_energies = None
        self.final_eigenvectors = None
        
    def setup_molecule(self):
        """Setup molecule and mean-field calculation"""
        if callable(self.mol):
            self.mol = self.mol(0, 0, self.basis)
        
        if self.output_file:
            self.mol.output = self.output_file
        self.mol.verbose = self.verbose
        self.mol.build()
        
        self.mf = scf.RHF(self.mol).run()
        print(f"RHF energy: {self.mf.e_tot}")
        
    def setup_lasscf_reference(self, mo_guess=None):
        """
        Setup reference LASSCF calculation with initial state averaging
        
        Args:
            mo_guess: initial molecular orbital guess (optional)
        """
        # Initial LASSCF with basic state averaging
        self.las = LASSCF(self.mf, self.ncas_sub, self.nelec_sub)
        self.las = self.las.state_average([0.5, 0.5],
            spins=[[1, -1], [-1, 1]],
            smults=[[2, 2], [2, 2]],    
            charges=[[0, 0], [0, 0]])
        
        if mo_guess is not None:
            mo = mo_guess
        else:
            # Default MO sorting and localization
            mo = self.las.sort_mo([16, 18, 22, 23, 24, 26])
            mo = self.las.localize_init_guess((list(range(5)), list(range(5, 10))), mo)
        
        self.las.kernel(mo)
        print(f"LASSCF({self.ncas_sub},{self.ncas_sub}) energy = {self.las.e_tot}")
        
        # Save reference orbitals
        if self.output_file:
            molden_file = self.output_file.replace('.log', '_lasscf66_631g.molden')
            molden.from_lasscf(self.las, molden_file)
    
    def setup_casci_reference(self):
        """Setup CASCI reference for comparison"""
        ncas_total = sum(self.ncas_sub)
        nelec_total = sum([sum(ne) for ne in self.nelec_sub])
        
        self.mc_casci = mcscf.CASCI(self.mf, ncas_total, nelec_total).set(
            fcisolver=csf_solver(self.mol, smult=1))
        self.mc_casci.kernel(self.las.mo_coeff)
        print(f"CASCI({ncas_total},{nelec_total}) energy = {self.mc_casci.e_tot}")
        
        # Save CASCI orbitals
        if self.output_file:
            molden_file = self.output_file.replace('.log', '_casscf66_631g.molden')
            molden.from_mcscf(self.mc_casci, molden_file, cas_natorb=True)
    
    def setup_charge_transfer_states(self):
        """
        Setup extended state averaging including charge transfer states
        """
        # Extended state averaging with charge transfer states
        self.las2 = self.las.state_average([0.5, 0.5, 0, 0],
            spins=[[1, -1], [-1, 1], [0, 0], [0, 0]],
            smults=[[2, 2], [2, 2], [1, 1], [1, 1]],    
            charges=[[0, 0], [0, 0], [-1, 1], [1, -1]])
        
        self.las2.lasci()
        self.las2.dump_spaces()
        
        # Get effective Hamiltonians and energies
        h2eff_sub, veff = self.las2.kernel(self.las.mo_coeff)[-2:]
        self.e_states = self.las2.e_states
        
        print("Charge transfer states setup completed")
        return self.e_states
    
    def compute_effective_hamiltonian(self):
        """
        Compute effective Hamiltonian matrix elements between LAS states
        """
        ncore, ncas = self.las2.ncore, self.las2.ncas
        nocc = ncore + ncas
        mo_coeff = self.las2.mo_coeff
        mo_core = mo_coeff[:, :ncore]
        mo_cas = mo_coeff[:, ncore:nocc]
        
        # Get core energy and integrals
        h2eff_sub, veff = self.las2.kernel(self.las.mo_coeff)[-2:]
        e0 = (self.las2._scf.energy_nuc() + 
              2 * (((self.las2._scf.get_hcore() + veff.c/2) @ mo_core) * mo_core).sum())
        
        h1 = mo_cas.conj().T @ (self.las2._scf.get_hcore() + veff.c) @ mo_cas
        h2 = h2eff_sub[ncore:nocc].reshape(ncas*ncas, ncas * (ncas+1) // 2)
        h2 = lib.numpy_helper.unpack_tril(h2).reshape(ncas, ncas, ncas, ncas)
        
        # Get electron configurations for each state
        nelec_fr = []
        for fcibox, nelec in zip(self.las2.fciboxes, self.las2.nelecas_sub):
            ne = sum(nelec)
            nelec_fr.append([_unpack_nelec(fcibox._get_nelec(solver, ne)) 
                           for solver in fcibox.fcisolvers])
        
        print("nelec_fr =", nelec_fr)
        
        # Compute effective Hamiltonian matrix elements
        self.ham_eff, self.s2_eff, self.ovlp_eff = self._slow_ham(
            self.las2.mol, h1, h2, self.las2.ci, self.las2.ncas_sub, nelec_fr)
        
        print("ham_eff\n", self.ham_eff)
        print("\novlap_eff\n", self.ovlp_eff)
        print("Convergence check:", self.las.converged, 
              self.e_states - (e0 + np.diag(self.ham_eff)))
        
        return self.ham_eff, self.ovlp_eff
    
    def build_vqe_wavefunctions(self, max_cycle=30, conv_tol=1e-6):
        """
        Build and optimize VQE wavefunctions for each LAS state using UCCSD parameterization
        
        Args:
            max_cycle: Maximum number of VQE optimization cycles
            conv_tol: Convergence tolerance for VQE optimization
        """
        nstates = 4
        self.psis = []
        self.vqe_energies = []
        ncas = sum(self.las2.ncas_sub)
        nelecas = sum([sum(ne) for ne in self.nelec_sub])
        
        # Get electron configurations
        nelec_fr = []
        for fcibox, nelec in zip(self.las2.fciboxes, self.las2.nelecas_sub):
            ne = sum(nelec)
            nelec_fr.append([_unpack_nelec(fcibox._get_nelec(solver, ne)) 
                           for solver in fcibox.fcisolvers])
        
        print("Building and optimizing VQE wavefunctions for", nstates, "states")
        print(f"VQE settings: max_cycle={max_cycle}, conv_tol={conv_tol}")
        
        for i in range(nstates):
            print(f"\n--- Optimizing State {i} ---")
            
            # Extract CI coefficients for state i
            ci = [[self.las2.ci[fragj][i]] for fragj in range(len(self.las2.ci))]
            nelec_sub = [nelec_fr[fragj][i] for fragj in range(len(nelec_fr))]
            nelecas_sub = [sum(nelec) for nelec in nelec_sub]
            
            print(f"State {i}: nelec_sub = {nelec_sub}, nelecas = {nelecas_sub}")
            
            # Create temporary LASSCF object for gradient calculation
            from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
            tmplas = LASSCF(self.mf, self.las2.ncas_sub, nelec_sub)
            tmplas.mo_coeff = self.las2.mo_coeff
            tmplas.ci = ci
            
            # Get gradient information for VQE optimization
            try:
                all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(tmplas, 1e-5)
                print(f"Selected amplitudes: {len(a_idxs_selected)} doubles, {len(i_idxs_selected)} singles")
            except Exception as e:
                print(f"Warning: Gradient calculation failed: {e}")
                a_idxs_selected, i_idxs_selected = [], []
            
            # Setup VQE solver for this state
            mc_uscc = mcscf.CASCI(self.mf, ncas, nelecas)
            mc_uscc.mo_coeff = self.las2.mo_coeff
            mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(self.mol, a_idxs_selected, i_idxs_selected)
            mc_uscc.fcisolver.norb_f = self.las2.ncas_sub
            
            # Set VQE optimization parameters
            lasci_ominus1.GLOBAL_MAX_CYCLE = max_cycle
            if hasattr(mc_uscc.fcisolver, 'conv_tol'):
                mc_uscc.fcisolver.conv_tol = conv_tol
            
            # Run VQE optimization
            print(f"Running VQE kernel optimization...")
            ci0 = self._cilas2f(ci, self.las2.ncas_sub, nelec_sub)
            e_vqe = mc_uscc.kernel(ci0=ci0)[0]
            
            # Store optimized wavefunction and energy
            self.psis.append(mc_uscc.fcisolver.psi)
            self.vqe_energies.append(e_vqe)
            
            print(f"State {i} VQE energy: {e_vqe:.8f} hartree")
        
        # Store the last mc_uscc for matrix element calculations
        self.mc_uscc = mc_uscc
        
        print(f"\nVQE optimization completed for {len(self.psis)} states")
        print("VQE energies:", self.vqe_energies)
        return self.psis
    
    def compute_vqe_matrix_elements(self):
        """
        Compute Hamiltonian and overlap matrix elements between VQE states
        """
        if self.psis is None:
            raise ValueError("VQE wavefunctions not built. Call build_vqe_wavefunctions() first.")
        
        # Get effective Hamiltonians
        h1eff, e_core = self.mc_uscc.get_h1eff(self.mc_uscc.mo_coeff)
        h2eff = self.mc_uscc.get_h2eff()
        h = [e_core, h1eff, h2eff]
        
        print("h1eff = \n", h1eff)
        print("h2eff shape:", h2eff.shape)
        
        nc = len(self.psis)
        self.S_matrix = np.zeros((nc, nc), dtype=np.complex128)
        self.H_matrix = np.zeros((nc, nc), dtype=np.complex128)
        
        # Compute all matrix elements
        for i in range(nc):
            for j in range(nc):
                self.S_matrix[i, j], self.H_matrix[i, j] = self._get_Sij_Hij(
                    self.psis[i], self.psis[j], h)
        
        print("S matrix:")
        self._print_complex_matrix(self.S_matrix)
        print("H matrix:")
        self._print_complex_matrix(self.H_matrix)
        
        return self.H_matrix, self.S_matrix
    
    def solve_generalized_eigenvalue_problem(self):
        """
        Solve generalized eigenvalue problem H|c> = E S|c>
        """
        if self.H_matrix is None or self.S_matrix is None:
            raise ValueError("Matrix elements not computed. Call compute_vqe_matrix_elements() first.")
        
        self.final_energies, self.final_eigenvectors = eigh(self.H_matrix, self.S_matrix)
        
        print("VQE-LASSI final energies:", self.final_energies)
        print("Ground state eigenvector:", self.final_eigenvectors[:, 0])
        
        return self.final_energies, self.final_eigenvectors
    
    def compare_with_lassi(self):
        """
        Compare results with standard LASSI calculation
        """
        if self.las2 is None:
            raise ValueError("LAS calculation not performed. Call setup_charge_transfer_states() first.")
        
        self.lsi = lassi.LASSI(self.las2)
        e_roots, si_vectors = self.lsi.kernel()
        
        print("Standard LASSI energies:", e_roots)
        print("Standard LASSI ground state vector:", si_vectors[:, 0])
        
        if self.final_energies is not None:
            print("\nEnergy comparison:")
            print("VQE-LASSI:", self.final_energies)
            print("Standard LASSI:", e_roots)
            print("Difference (VQE - LASSI):", self.final_energies - e_roots)
        
        return e_roots, si_vectors
    
    def run_full_calculation(self, mo_guess=None, vqe_max_cycle=5, vqe_conv_tol=1e-6):
        """
        Run the complete LASSI-VQE calculation
        
        Args:
            mo_guess: initial molecular orbital guess (optional)
            vqe_max_cycle: maximum VQE optimization cycles (default: 30)
            vqe_conv_tol: VQE convergence tolerance (default: 1e-6)
            
        Returns:
            tuple: (final_energies, final_eigenvectors, lassi_energies)
        """
        print("=== Starting LASSI-VQE Calculation ===")
        
        # Step 1: Setup molecule and mean-field
        print("\n1. Setting up molecule and mean-field...")
        self.setup_molecule()
        
        # Step 2: Setup reference LASSCF
        print("\n2. Setting up reference LASSCF...")
        self.setup_lasscf_reference(mo_guess)
        
        # Step 3: Setup CASCI reference
        print("\n3. Setting up CASCI reference...")
        self.setup_casci_reference()
        
        # Step 4: Setup charge transfer states
        print("\n4. Setting up charge transfer states...")
        self.setup_charge_transfer_states()
        
        # Step 5: Compute effective Hamiltonian
        print("\n5. Computing effective Hamiltonian...")
        self.compute_effective_hamiltonian()
        
        # Step 6: Build VQE wavefunctions
        print("\n6. Building VQE wavefunctions...")
        self.build_vqe_wavefunctions(max_cycle=vqe_max_cycle, conv_tol=vqe_conv_tol)
        
        # Step 7: Compute VQE matrix elements
        print("\n7. Computing VQE matrix elements...")
        self.compute_vqe_matrix_elements()
        
        # Step 8: Solve eigenvalue problem
        print("\n8. Solving generalized eigenvalue problem...")
        self.solve_generalized_eigenvalue_problem()
        
        # Step 9: Compare with standard LASSI
        print("\n9. Comparing with standard LASSI...")
        lassi_energies, _ = self.compare_with_lassi()
        
        print("\n=== LASSI-VQE Calculation Complete ===")
        
        return self.final_energies, self.final_eigenvectors, lassi_energies
    
    def save_molden_files(self, base_name=None):
        """Save molecular orbital files"""
        if base_name is None:
            base_name = self.output_file.replace('.log', '') if self.output_file else 'lassi_vqe'
        
        if self.lsi is not None:
            molden.from_lassi(self.lsi, f'{base_name}_lassi.molden', state=0)
        
    # Helper methods (private)
    def _addr_outer_product(self, norb_f, nelec_f):
        """Compute outer product addresses for CI vectors"""
        norb = sum(norb_f)
        nelec = sum(nelec_f)
        norbrange = np.cumsum(norb_f)
        addrs = []
        for i in range(0, len(norbrange)):
            new_addrs = (cistring.sub_addrs(norb, nelec, 
                        range(norbrange[i]-norb_f[i], norbrange[i]), nelec_f[i]) 
                        if nelec_f[i] else [])
            if len(addrs) == 0:
                addrs = new_addrs
            elif len(new_addrs) > 0:
                addrs = np.intersect1d(addrs, new_addrs)
        return addrs

    def _ci_outer_product(self, ci_f, norb_f, nelec_f):
        """Compute outer product of CI vectors"""
        ci_dp = ci_f[-1].copy()
        for ci_r in ci_f[-2::-1]:
            ndeta_1, ndetb_1 = ci_dp.shape
            ndeta_2, ndetb_2 = ci_r.shape
            ci_dp = np.multiply.outer(ci_dp, ci_r)
            ci_dp = ci_dp.transpose(0, 2, 1, 3).reshape(ndeta_1*ndeta_2, ndetb_1*ndetb_2)
        
        neleca_f = [ne[0] for ne in nelec_f]
        nelecb_f = [ne[1] for ne in nelec_f]
        addrs_a = self._addr_outer_product(norb_f, neleca_f)
        addrs_b = self._addr_outer_product(norb_f, nelecb_f)
        
        ci = np.zeros((cistring.num_strings(sum(norb_f), sum(neleca_f)), 
                      cistring.num_strings(sum(norb_f), sum(nelecb_f))),
                     dtype=ci_dp.dtype)
        ci[np.ix_(addrs_a, addrs_b)] = ci_dp[:, :] / linalg.norm(ci_dp)
        return ci

    def _slow_ham(self, mol, h1, h2, ci_fr, norb_f, nelec_fr, orbsym=None):
        """Compute effective Hamiltonian matrix elements"""
        ci, nelec = self._ci_outer_product_states(ci_fr, norb_f, nelec_fr)
        solver = fci.solver(mol).set(orbsym=orbsym)
        norb = sum(norb_f)
        h2eff = solver.absorb_h1e(h1, h2, norb, nelec, 0.5)
        ham_ci = [solver.contract_2e(h2eff, c, norb, nelec) for c in ci]
        s2_ci = [contract_ss(c, norb, nelec) for c in ci]
        ham_eff = np.array([[c.ravel().dot(hc.ravel()) for hc in ham_ci] for c in ci])
        s2_eff = np.array([[c.ravel().dot(s2c.ravel()) for s2c in s2_ci] for c in ci])
        ovlp_eff = np.array([[bra.ravel().dot(ket.ravel()) for ket in ci] for bra in ci])
        return ham_eff, s2_eff, ovlp_eff

    def _ci_outer_product_states(self, ci_fr, norb_f, nelec_fr):
        """Compute outer product for all states"""
        ci_r = []
        for state in range(len(ci_fr[0])):
            ci_f = [ci[state] for ci in ci_fr]
            nelec_f = [nelec[state] for nelec in nelec_fr]
            ci_r.append(self._ci_outer_product(ci_f, norb_f, nelec_f))
        nelec = (sum([ne[0] for ne in nelec_f]),
                sum([ne[1] for ne in nelec_f]))
        return ci_r, nelec

    def _get_Sij_Hij(self, psi_i, psi_j, h):
        """Compute overlap and Hamiltonian matrix elements between VQE states"""
        ucj, hucj = psi_j.hc_x(psi_j.x, h)[1:3]
        uci, huci = psi_i.hc_x(psi_i.x, h)[1:3]
        ucj, hucj = ucj.ravel(), hucj.ravel()
        uci = uci.ravel()
        Sij = uci.conj().dot(ucj)
        Hij = uci.conj().dot(hucj)
        return Sij, Hij

    def _cilas2f(self, lasci, norb_f, nelec_f):
        """Convert LAS CI to Fock space representation"""
        ci_f = []
        for i, ci in enumerate(lasci):
            ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
        return ci_f

    def _print_complex_matrix(self, mat):
        """Print complex matrix in a readable format"""
        for row in mat:
            print("  ".join(f"{x.real:+.17f}{x.imag:+.17f}j" 
                           if x.imag != 0 else f"{x.real:+.17f}+{x.imag:.17f}j" 
                           for x in row))