import copy
import numpy as np
from mrh.exploratory.unitary_cc.lasuccsd import gen_usccsd_op
from mrh.exploratory.citools import lasci_ominus1, fockspace
from pyscf import lib
import time
import functools
from helper.util import timeit, print_list_matrix, get_sorted_excitations
from helper import op

from scipy.linalg import eigh

def noci_symmetric_orth(H, S, rcond=1e-10):
    # eigen-decompose S
    w, U = eigh(S)
    # sort descending
    idx = np.argsort(w)[::-1]
    w, U = w[idx], U[:, idx]

    # keep well-conditioned subspace
    wmax = w[0]
    keep = w > (rcond * wmax)
    Uk = U[:, keep]
    wk = w[keep]

    X = Uk / np.sqrt(wk)  # columns scaled => Uk @ diag(wk^-1/2)
    Ht = X.conj().T @ H @ X
    e, y = eigh(Ht)
    c = X @ y
    return e, c, keep, w


def jordan_wigner_res(ops, nspin_orbs):
    # apply jordan-wigner transformation implicitly to get the final per-site result
    # input: ops = (('annihilate', 7), ('annihilate', 5), ('create', 7), ('create', 7))
    # output: res_vaccum and res_occupied, are list of length nspin_orbs with abs values 1 (unoccupied), 2 (occupied), 3 (vanish), and signs for the phase
    # for example res_vaccum[3] = -2 means that when applying the jordan-wigner single site operators on |0>_3, we obtain -|1>_3
    res_vaccum = [1 for _ in range(nspin_orbs)] # 1 -> unoccupied, 2 -> occupied, +- sign, +-3 -> vanish
    res_occupied =[2 for _ in range(nspin_orbs)]
    for op_type, op_idx in ops:
        for res in [res_vaccum, res_occupied]:
            
            if op_type == "identity":
                continue
        
            elif op_type == "number":
                if abs(res[op_idx]) == 1:
                    res[op_idx] = 3
            else:
                # annihilation or creation
                for k in range(op_idx):
                    # s0,..,sk-1
                    phase = -1 if abs(res[k]) == 2 else 1
                    res[k] *= phase
                # bk
            
                if abs(res[op_idx]) == 2:
                    if op_type == "annihilate":
                        res[op_idx] /= 2
                    else:
                        res[op_idx] = 3
                elif abs(res[op_idx]) == 1:
                    if op_type == "create":
                        res[op_idx] *= 2
                    else:
                        res[op_idx] = 3
    
    return res_vaccum, res_occupied

def binarr(index, length):
    return [int(b) for b in format(index, f'0{length}b')]

def las_a_las_jwres(jwres_vaccum, jwres_occupied, ci_fs, frag_sorbs):
    # evaluate the <LAS| aiajak'am... |LAS>
    # input: jwres_vaccum and jwres_occupied are jw representations of operators
    # ci_f is the fragmented CI vector 
    inner_prod = 1
    for i, ci_f in enumerate(ci_fs):
        new_ci_f = np.zeros_like(ci_f.ravel())
        for det_idx, amp in enumerate(ci_f.ravel()):
            if abs(amp) < 1e-5:
                continue
            binary_array = binarr(det_idx , len(frag_sorbs[i]))
            res_bin_array = np.zeros_like(binary_array)
            coef = amp
            for forb_idx, occ in enumerate(binary_array[::-1]):
                orb_idx = frag_sorbs[i][forb_idx]
                if occ == 0:
                    res = jwres_vaccum[orb_idx]
                else:
                    res = jwres_occupied[orb_idx]
                if res == 3:
                    coef = 0
                    break
                if abs(res) == 1:
                    res_bin_array[forb_idx] = 0
                elif abs(res) == 2:
                    res_bin_array[forb_idx] = 1
                coef *= -1 if res < 0 else 1
            
            res_det_idx = int(''.join(map(str, res_bin_array[::-1])), 2)
            new_ci_f[res_det_idx] += coef
    
        inner_prod *= np.vdot(ci_f.ravel(), new_ci_f.ravel())
        if abs(inner_prod) < 1e-5:
            return 0
    return inner_prod

class LASSolver_LSCC():

    def __init__(self, las, a_idxs, i_idxs, frag_sorbs):
        self.las = las
        # add the first excitation to be identity operator 
        self.a_idxs = a_idxs
        self.i_idxs = i_idxs
        self.frag_sorbs = frag_sorbs
        self.nsorb = max([max(frag_sorb) for frag_sorb in self.frag_sorbs]) + 1

    def kernel(self, h, norb_f, nelec_f, ci0_f):

        Hop = op.get_hop(self.las)

        Aops = [op.IdentityOp()] 
        for a_idx, i_idx in zip(self.a_idxs, self.i_idxs):
            Aterms = [
                (1, [("annihilate",i ) for i in i_idx] + [("create",a) for a in a_idx[::-1]]),
                (-1,[("annihilate",a) for a in a_idx] + [("create", i) for i in i_idx[::-1]]),
            ]
            Aops.append(op.Op(Aterms))

        S = np.zeros((len(Aops), len(Aops)), dtype=np.complex128)
        for i, Ai in enumerate(Aops):
            for j, Aj in enumerate(Aops):
                Aid_AjOp = Ai.dagger() * Aj
                Sij = 0
                for term in Aid_AjOp.terms:
                    coef = term[0]
                    Aterm = term[1]
                    jwop_vaccum, jwop_occupied = jordan_wigner_res(Aterm,   self.nsorb) # it only has one term
                    Sij += coef * las_a_las_jwres(jwop_vaccum, jwop_occupied, ci0_f, self.frag_sorbs)
                S[i,j] = Sij
                                                                
        H = np.zeros((len(Aops), len(Aops)), dtype=np.complex128)
        
        for i, Ai in enumerate(Aops):
            for j, Aj in enumerate(Aops):   
                # for future optimization we can only evaluate upper triangular part
                Aid_H_AjOp = Ai.dagger() * Hop * Aj
                hij = 0
                for term in Aid_H_AjOp.terms:
                    # here many terms may not preserve particle number in fragment,
                    # for future optimization we can skip those terms 
                    coef = term[0]
                    Aterm = term[1]
                    jwop_vaccum, jwop_occupied = jordan_wigner_res(Aterm, self.nsorb)
                    hij += coef * las_a_las_jwres(jwop_vaccum, jwop_occupied, ci0_f, self.frag_sorbs)
                H[i,j] = hij
        
        # solve the generalized eigenvalue problem
        e, c, keep, w = noci_symmetric_orth(H, S)
        idx = np.argmin(e)
        return e[idx].real, c[:,idx]
    

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
    from helper import op
    from pathlib import Path
    from time import time
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
    mc_uscc = mcscf.CASCI(mf, sum(ncas_f), sum(nelecas_f))
    h1eff,e_core= mc_uscc.get_h1eff(mc_uscc.mo_coeff)
    print("Core energy from mc_uscc.get_h1eff: ", e_core)
    mc_uscc.mo_coeff = las.mo_coeff
    lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
    mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)
    mc_uscc.fcisolver.norb_f = [2,2]
    mc_uscc.kernel(ci0=las_ci0_f)
    print("Epsilon: {:.9f} | Number of parameters: {:.0f} | LASUSCCSD energy: {:.9f}".format(epsilon, len(a_idxs_selected), mc_uscc.e_tot))
    
    mc_uscc.fcisolver = FCISolver_CC(mol, a_idxs_selected, i_idxs_selected, t = np.pi / 2)

    mc_uscc.fcisolver.norb_f = ncas_f

    start_time = time()
    mc_uscc.kernel(ci0=las_ci0_f)
    print("LASLCCSD energy: {:.9f}".format(mc_uscc.e_tot))
    print("Time taken: {:.2f} seconds".format(time() - start_time))

    las_lscc_solver = LASSolver_LSCC(las, a_idxs_selected, i_idxs_selected, [[0,1,4,5], [2,3,6,7]])
    start_time = time()
    energy_lscc, c_lscc = las_lscc_solver.kernel(op.get_hop(las), ncas_f, nelecas_f, las_ci0_f)
    print("LAS-LSCCSD energy: {:.9f}".format(energy_lscc))
    print("Time taken: {:.2f} seconds".format(time() - start_time))
    # c = mc_uscc.fcisolver.psi0.dp_ci (las_ci0_f)
    # print("norm c = ", np.linalg.norm(c))