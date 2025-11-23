import copy
import numpy as np
from mrh.exploratory.unitary_cc.lasuccsd import gen_usccsd_op
from mrh.exploratory.citools import lasci_ominus1, fockspace
from pyscf import lib
import time
import functools
from helper.util import timeit, print_matrix, get_sorted_excitations


class FCISolver_CC(lasci_ominus1.FCISolver):

    def __init__(self, mol, a_idxs, i_idxs, linearize = True,  t = 200):
        super().__init__(mol)
        self.mol = mol
        # add the first excitation to be identity operator 
        self.a_idxs = a_idxs
        self.i_idxs = i_idxs
        self.linearize = linearize
        self.t = t

    def get_uop(self, norb, nlas, t1_s2sym=None):
        uop = gen_usccsd_op(norb, nlas, self.a_idxs, self.i_idxs)
        uop.linearize = self.linearize # 
        return uop
    
    def get_fcivec(self, si):
        fci = np.zeros_like(self.psi0.get_fcivec(), dtype=np.complex128)
        for i, c in enumerate(si):
            if c == 0: continue
            psi_i = copy.copy(self.psi0)
            psi_i.x[psi_i.nconstr + i] = self.t
            fci += c * psi_i.get_fcivec()
        
        return fci

    def get_uci_hci(self, i, h):
        self.psi0.x[self.psi0.nconstr + i] = self.t
        uci, huci = self.psi0.hc_x(self.psi0.x, h)[1:3]

        # reset the x vector
        self.psi0.x[self.psi0.nconstr + i] = 0
        return uci, huci

    def _get_Sij_Hij(self, i, j, h):
        uci, huci = self.get_uci_hci(i, h)
        ucj, hucj = self.get_uci_hci(j, h)
        uci = uci.ravel()
        return uci.conj().dot(ucj.ravel()), uci.conj().dot(hucj.ravel())

    def build_S_H(self, h):
        # self.psi0 = LASUCCTrialState (self, ci0_f, norb, norb_f, nelec)
        assert self.psi0 is not None, "LASUCCTrialState not initialized"

        n = len(self.a_idxs)
        # reuse existing S/H if present, otherwise create new ones; only compute additional elements
        S = np.zeros((n, n), dtype=np.complex128)
        H = np.zeros((n, n), dtype=np.complex128)

        old_S = getattr(self, "S", np.zeros((0, 0), dtype=np.complex128))
        old_H = getattr(self, "H", np.zeros((0, 0), dtype=np.complex128))

        m = old_S.shape[0]
        if m > 0:
            S[:m, :m] = old_S[:m, :m].copy()
            H[:m, :m] = old_H[:m, :m].copy()
        for i in range(m, n):
            for j in range(i, n):
                sij, hij = self._get_Sij_Hij(i, j, h)
                S[i, j] = sij
                H[i, j] = hij
                if i != j:
                    S[j, i] = np.conj(sij)
                    H[j, i] = np.conj(hij)

        self.S, self.H = S, H

    def select_psi(self, cond_thresh=8e5):
        n = self.S.shape[0]
        select_idx = np.ones(n, dtype=bool)
        for i in range(1, n + 1):
            Si = self.S[:i, :i]
            try:
                cnum = np.linalg.cond(Si)
            except Exception:
                cnum = float('inf')
            if cnum > cond_thresh:
                if self.log.verbose >= 4:
                    self.log.debug("discarding psi[{}] | excitation = {}".format(i - 1, (self.a_idxs[i - 1], self.i_idxs[i - 1])) )
                # print("discarding psi[{}] | condition number = {}".format(i - 1, cnum) )
                select_idx[i - 1] = 0
        return select_idx

    def kernel(self, h1, h2, norb, nelec, norb_f=None, ci0 = None, ci0_f=None,
            tol=1e-8, gtol=1e-6, max_cycle=None, 
            orbsym=None, wfnsym=None, ecore=0, **kwargs):
        
        if norb_f is None: norb_f = self.norb_f

        ci0_f = ci0 # my change
        if ci0_f is None: ci0_f = lasci_ominus1.get_init_guess (self, norb, nelec, norb_f, h1, h2)
        verbose = kwargs.get ('verbose', 1)
        cond_thresh = kwargs.get('cond_thresh', 8e5)
        self.psi0 = lasci_ominus1.LASUCCTrialState (self, ci0_f, norb, norb_f, nelec)
        if isinstance (verbose, lib.logger.Logger):
            self.log = verbose
            verbose = self.log.verbose
        else:
            self.log = lib.logger.new_logger (fci, verbose)
        
        h = [ecore, h1, h2]
        self.build_S_H(h)
        idx_sel = self.select_psi(cond_thresh)
        S_sel = self.S[np.ix_(idx_sel, idx_sel)]
        H_sel = self.H[np.ix_(idx_sel, idx_sel)]
        eigvals, eigvecs = np.linalg.eig (np.linalg.solve (S_sel, H_sel))
        e_tot = np.min (eigvals).real
        si_sel = eigvecs[:, np.argmin (eigvals)]
        si = np.zeros (self.S.shape[0], dtype=np.complex128)
        si[idx_sel] = si_sel
        self.si = si.real
        ci = self.get_fcivec (si)
        self.ci = ci
        return e_tot, ci


if __name__ == '__main__':
    import numpy as np
    import pyscf
    from pyscf import gto, scf, lib, mcscf
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    from mrh.exploratory.unitary_cc import lasuccsd
    from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
    from mrh.exploratory.citools import grad, lasci_ominus1
    from helper.util import print_list_matrix, get_sorted_excitations
    # specify the hyperparameters
    ncas_f = (2,2)
    nelecas_f = (2,2)
    spin_sub_f = (1,1)
    epsilon = 0.02
    frag_atom_list=((0, 1), (2, 3))

    # Initializing the molecule with RHF
    #===================================
    xyz = '''H      0.000000000000   0.000000000000   0.000000000000
    H      1.000000000000   0.000000000000   0.000000000000
    H      0.273746762116   2.195450598147   0.100000000000
    H      1.232912762116   1.895450598147  -0.100000000000'''
    mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log',
        verbose=0)
    mf = scf.RHF (mol).run ()
    
    print ("RHF energy = ", mf.e_tot)

    # Running LASSCF
    #===================================
    las = LASSCF (mf, ncas_f, nelecas_f, spin_sub=spin_sub_f)
    las.verbose = 4
    mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
    las.kernel (mo_loc)
    print ("LASSCF energy = ", las.e_tot)
    las_ci0_f = cilas2f(las.ci, ncas_f, nelecas_f)

    # Run CASCI using molecular orbitals from LASSCF
    cas = mcscf.CASCI (mf, sum(ncas_f), sum(nelecas_f))
    cas.mo_coeff = las.mo_coeff
    cas.kernel()
    print ("CASCI energy = ", cas.e_tot)

    #Getting gradient for all cluster excitations through LAS-UCCSD gradients, may use your desired epsilon for selection
    #====================================================================================================================
    a_idxs_selected, i_idxs_selected = get_sorted_excitations(las, epsilon=epsilon)
    excitations = []
    for idx,(a, i) in enumerate(zip(a_idxs_selected, i_idxs_selected)):
        print(f"Selected excitation {idx}: a = {a}, i = {i}")
    

    #Computing energy through the LAS-UCC kernel using selected excitations
    #==========================================================================================
    mc_uscc = mcscf.CASCI(mf, sum(ncas_f), sum(nelecas_f))
    mc_uscc.mo_coeff = las.mo_coeff
    lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
    mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
    mc_uscc.fcisolver.norb_f = ncas_f
    mc_uscc.kernel(ci0=las_ci0_f)
    print("Epsilon: {:.9f} | Number of parameters: {:.0f} | LASUSCCSD energy: {:.9f}".format(epsilon, len(a_idxs_selected), mc_uscc.e_tot))
    print("final vqe amplitudes: \n", mc_uscc.fcisolver.psi.x)

    # include identity operator, notice that we have to insert it at front otherwise it might get filtered out
    a_idxs_selected.insert(0, np.array([0,1], dtype=np.uint8))
    i_idxs_selected.insert(0, np.array([0,1], dtype=np.uint8))

    mc_uscc.fcisolver = FCISolver_CC(mol, a_idxs_selected, i_idxs_selected, t = 1000)
    mc_uscc.fcisolver.norb_f = ncas_f
    mc_uscc.kernel(ci0=las_ci0_f, verbose=4)
    print("LASLCCSD energy: {:.9f}".format(mc_uscc.e_tot))
    print("final lcc amplitudes: \n", mc_uscc.fcisolver.si)