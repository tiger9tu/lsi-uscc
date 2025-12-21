import copy
import numpy as np
from mrh.exploratory.unitary_cc.lasuccsd import gen_usccsd_op
from mrh.exploratory.citools import lasci_ominus1, fockspace
from pyscf import lib
import time
import functools
from helper.util import timeit, print_list_matrix, get_sorted_excitations


class FCISolver_CC(lasci_ominus1.FCISolver):

    def __init__(self, mol, a_idxs, i_idxs, ci0f_s = None ,  t = np.pi / 2):
        super().__init__(mol)
        self.mol = mol
        # add the first excitation to be identity operator 
        self.a_idxs = a_idxs
        self.i_idxs = i_idxs
        # self.linearize = linearize
        self.t = t
        self.ci0_fs = ci0f_s
        self.h = None

    def get_uop(self, norb, nlas, t1_s2sym=None):
        uop = gen_usccsd_op(norb, nlas, self.a_idxs, self.i_idxs)
        # uop.linearize = self.linearize # 
        return uop
    
    def get_fcivec(self, lccsi):
        fci = np.zeros_like(self.las_psi0s[0].get_fcivec(), dtype=np.complex128)
        
        for i, c in enumerate(lccsi):
            if c == 0: continue
            uilsi = self.get_uilsi_huilsi(i)[0]
            fci += c * uilsi
        
        return fci

    # def get_uci_hci(self, i, h):
    #     self.psi0.x[self.psi0.nconstr + i] = self.t
    #     uci, huci = self.psi0.hc_x(self.psi0.x, h)[1:3]

    #     # reset the x vector
    #     self.psi0.x[self.psi0.nconstr + i] = 0
    #     return uci, huci

    def get_uilsi_huilsi(self, i):
        uilass = []
        huilass = []
        for psi0 in self.las_psi0s:
            psi0.x[psi0.nconstr + i] = self.t
            uilas, huilas = psi0.hc_x(psi0.x, self.h)[1:3]
            uilass.append(uilas)
            huilass.append(huilas)
            # reset the x vector
            psi0.x[psi0.nconstr + i] = 0
        uilsi = sum(self.si[i] * uilass[i] for i in range(len(self.si)))
        huilsi = sum(self.si[i] * huilass[i] for i in range(len(self.si))) 
        return uilsi, huilsi   

    def _get_Sij_Hij(self, i, j):
        uilsi, huilsi = self.get_uilsi_huilsi(i)
        ulsi, hulsi = self.get_uilsi_huilsi(j)
        uilsi = uilsi.ravel()
        return uilsi.conj().dot(ulsi.ravel()), uilsi.conj().dot(hulsi.ravel())

    def build_S_H(self):
        # self.psi0 = LASUCCTrialState (self, ci0_f, norb, norb_f, nelec)
        assert self.las_psi0s is not None, "LASUCCTrialState not initialized"
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
                sij, hij = self._get_Sij_Hij(i, j)
                S[i, j] = sij
                H[i, j] = hij
                if i != j:
                    S[j, i] = np.conj(sij)
                    H[j, i] = np.conj(hij)

        self.S, self.H = S, H

    def select_ui(self, cond_thresh=8e5):
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

    def kernel(self, h1, h2, norb, nelec, norb_f=None, ci0=None, ci0_f=None, # lassi compatible
            tol=1e-8, gtol=1e-6, max_cycle=None, 
            orbsym=None, wfnsym=None, ecore=0, **kwargs):
        
        if norb_f is None: norb_f = self.norb_f
        cond_thresh = kwargs.get('cond_thresh', 8e5)
        verbose = kwargs.get ('verbose', 1)
        if self.ci0_fs is None:
            if ci0 is None:
                print("Warning: ci0_fs is None, generating initial guess using LASCI")
                ci0 = lasci_ominus1.get_init_guess (self, norb, nelec, norb_f, h1, h2)
            self.ci0_fs = [ci0]
            self.si = [1.0]
        else:
            assert self.si is not None, "si vector not provided for multiple ci0_fs"
        
        self.las_psi0s = []
        for ci0_f in self.ci0_fs:
            self.las_psi0s.append(lasci_ominus1.LASUCCTrialState (self, ci0_f, norb, norb_f, nelec))
        
        if isinstance (verbose, lib.logger.Logger):
            self.log = verbose
            verbose = self.log.verbose
        else:
            self.log = lib.logger.new_logger (fci, verbose)
        
        self.h = [ecore, h1, h2]
        self.build_S_H()
        idx_sel = self.select_ui(cond_thresh)
        S_sel = self.S[np.ix_(idx_sel, idx_sel)]
        H_sel = self.H[np.ix_(idx_sel, idx_sel)]
        eigvals, eigvecs = np.linalg.eig (np.linalg.solve (S_sel, H_sel))
        e_tot = np.min (eigvals).real
        lccsi_sel = eigvecs[:, np.argmin (eigvals)]
        lccsi = np.zeros (self.S.shape[0], dtype=np.complex128)
        lccsi[idx_sel] = lccsi_sel
        self.lccsi = lccsi.real
        ci = self.get_fcivec (lccsi)
        self.ci = ci # fci
        return e_tot, np.asarray(ci.real, dtype=np.float64)


if __name__ == '__main__':
    import numpy as np
    import pyscf
    from pyscf import gto, scf, lib, mcscf
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    from mrh.exploratory.unitary_cc import lasuccsd
    from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
    from mrh.exploratory.citools import grad, lasci_ominus1
    from lcc.lcc_solver import FCISolver_CC
    from helper.util import print_list_matrix, get_sorted_excitations, cilas2f

    from pathlib import Path

    # pwd = Path(__file__).resolve().parent
    # geom_path = pwd.parent / 'geom' / 'h4.xyz'
    # with geom_path.open('r') as f:
    #     xyz = f.read()
    xyz = '''H      0.000000000000   0.000000000000   0.000000000000
    H      1.000000000000   0.000000000000   0.000000000000
    H      0.273746762116   2.195450598147   0.100000000000
    H      1.232912762116   1.895450598147  -0.100000000000'''
    # Initializing the molecule with RHF
    #===================================
    ncas_f = (2,2)
    nelecas_f = (2,2)
    spin_sub_f = (1,1)
    frag_atom_list = ((0,1),(2,3))

    mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log',
        verbose=0)
    mf = scf.RHF (mol).run ()
    ref = mcscf.CASCI (mf, 4, 4).run () # = FCI
    print ("RHF energy = ", mf.e_tot)
    print ("CASCI energy = ", ref.e_tot)


    # Running LASSCF
    #===================================
    las = LASSCF (mf, ncas_f, nelecas_f, spin_sub=spin_sub_f)
    las.verbose = 4

    mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
    las.kernel (mo_loc)
    print ("LASSCF energy = ", las.e_tot)
    las_ci0_f = cilas2f(las.ci, ncas_f, nelecas_f)
    #Getting gradient for all cluster excitations through LAS-UCCSD gradients, may use your desired epsilon for selection
    #====================================================================================================================

    epsilon = 0.001
    all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(las, epsilon)
    # sort the selected excitations by gradiDent magnitude
    gredients = np.array(g_sel)[:,0]
    sorted_indices = np.argsort(-np.abs(gredients))
    a_idxs_selected = [a_idxs_selected[i] for i in sorted_indices]
    i_idxs_selected = [i_idxs_selected[i] for i in sorted_indices]


    #Computing energy through the LAS-UCC kernel using selected excitations
    #==========================================================================================
    epsilon=0.01
    mc_uscc = mcscf.CASCI(mf, sum(ncas_f), sum(nelecas_f))
    mc_uscc.mo_coeff = las.mo_coeff
    lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
    mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
    mc_uscc.fcisolver.norb_f = [2,2]
    mc_uscc.kernel(ci0=las_ci0_f)
    print("Epsilon: {:.9f} | Number of parameters: {:.0f} | LASUSCCSD energy: {:.9f}".format(epsilon, len(a_idxs_selected), mc_uscc.e_tot))

    # print("a_idxs_selected = ", a_idxs_selected)
    # to include the Identity operator
    # a_idxs_selected = []
    # i_idxs_selected = []
    a_idxs_selected.insert(0, np.array([0], dtype=np.uint8))
    i_idxs_selected.insert(0, np.array([0], dtype=np.uint8))
    # print("a_idxs_selected = ", a_idxs_selected)
    # t does not matter now, just to avoid error
    mc_uscc.fcisolver = FCISolver_CC(mol, a_idxs_selected, i_idxs_selected, t = np.pi / 2)

    mc_uscc.fcisolver.norb_f = ncas_f

    mc_uscc.kernel(ci0=las_ci0_f)
    print("LASLCCSD energy: {:.9f}".format(mc_uscc.e_tot))

    # c = mc_uscc.fcisolver.psi0.dp_ci (las_ci0_f)
    # print("norm c = ", np.linalg.norm(c))