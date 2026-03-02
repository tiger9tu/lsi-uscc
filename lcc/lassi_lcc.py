from mrh.my_pyscf.lassi import LASSI
from mrh.my_pyscf.lassi.spaces import spin_shuffle, spin_shuffle_ci
from mrh.my_pyscf.mcscf.addons import state_average_n_mix, get_h1e_zipped_fcisolver
from mrh.my_pyscf.mcscf.productstate import ImpureProductStateFCISolver
import numpy as np
from pyscf.fci import addons
from pyscf.csf_fci import csf_solver
from helper.op_ci import apply_operator_string_fci
from helper import util
from copy import deepcopy

class LASSI_LSCC (LASSI):
    # my class for performing LSCC using LASSI implementation
    # This can be further optimized
    def __init__(self, las, a_idxs, i_idxs, frag_orbs, opt=1, **kwargs):
        # a_idxs and i_idxs are spinless creation and annihilation operators
        self.a_idxs = a_idxs
        self.i_idxs = i_idxs
        self.ci = las.ci

        LASSI.__init__(self, las, opt=opt, **kwargs)

    def getAci(self, a_idx, i_idx):
        # Get the CI vector for the state with excitation from i_idx to a_idx
        # A|psi> = a0a1...i1i0|psi>
        # we return A|psi>, A'|psi>
        
        frag_orbs_start = [0]
        for nelecf in self.ncas_sub[:-1]:
            frag_orbs_start.append(frag_orbs_start[-1] + nelecf)
        

        # we perform the operators by fragments
        # we ignore the sign factor, which is not important
        # since the indexing of all the operators differ, we can shuffle them freely
        # ci0 = deepcopy(self.ci0)
        Aci = deepcopy([frag_ci[0] for frag_ci in self.ci])
        frag_ops = [[] for _ in range(len(self.ci))]

        for op_type, idx_list in [('ann', i_idx), ('cre', a_idx[::-1])]:
            for idx in idx_list:
                spatial = idx % self.ncas
                spin = idx // self.ncas
                frag_idx = np.searchsorted(frag_orbs_start, spatial, side='right') - 1
                idx_in_frag = spatial - frag_orbs_start[frag_idx]
                frag_ops[frag_idx].append((op_type, idx_in_frag, spin))
        
        nelecas_sub = deepcopy(self.nelecas_sub)
        for i, ci_f in enumerate(Aci):
            if len(frag_ops[i]) == 0:
                continue
            ci_f_new, (neleca, nelecb) = apply_operator_string_fci(ci_f, self.ncas_sub[i], self.nelecas_sub[i], frag_ops[i])
            if ci_f_new is None or np.all(ci_f_new == 0):
                return None, None  # invalid excitation gives zero
            Aci[i] = ci_f_new
            nelecas_sub[i] = (neleca, nelecb)

        return Aci, nelecas_sub

    def get_e_states(self):
        las = self._las
        # h1eff, energy_core = las.h1e_for_cas (mo_coeff=mo_coeff, ncas=las.ncas, ncore=las.ncore)
        h1eff, energy_core = las.h1e_for_cas ()
        # eri_cas = las.get_h2cas (mo_coeff) 
        eri_cas = las.get_h2cas () 
        e_states = np.zeros(self.nroots)
        for state in range (self.nroots):
            fcisolvers = [b.fcisolvers[state] for b in self.fciboxes]
            ci0_i = [[c[state]] for c in self.ci]
            solver = ImpureProductStateFCISolver (fcisolvers)
            # norb_f and nelec_f are just the original ones, later it will be combined with fcisolver for
            # the perticular state to get the nelec for that state
            norb_f = las.ncas_sub
            nelec_f = las.nelecas_sub
            energy_elec = solver.energy_elec (h1eff, eri_cas, ci0_i, norb_f, nelec_f, ecore=energy_core)
            # e_cas[state] = energy_elec
            e_states[state] = energy_elec + energy_core

        print("e_states = ", e_states)
        return e_states

    def prepare_states_(self):
        # self.converged, las = self.prepare_states ()
        #self.__dict__.update(las.__dict__) # Unsafe
        self.fciboxes = [[] for _ in range(self.nfrags)]

        # def _get_csfsolver(nelecas_sub_fi):
        #     fcisolver = csf_solver(self.mol, smult=nelecas_sub_fi[0] - nelecas_sub_fi[1])
        #     fcisolver.nelec = nelecas_sub_fi
        #     fcisolver.norb = self.nelecas[0]  # Note: adjust index if needed
        #     fcisolver.spin = nelecas_sub_fi[0] - nelecas_sub_fi[1]
        #     fcisolver.smult = abs(fcisolver.spin) + 1
        #     return fcisolver

        # Add the original las state
        # fciboxes_list = [[_get_csfsolver(nelecas_sub_fi)] for nelecas_sub_fi in self.nelecas_sub]
        # nelecas_sub_buffer = [[] for _ in range(self.nfrags)]
        # for fi in range(self.nfrags):
        #      nelecas_sub_buffer[fi].append(deepcopy(self.nelecas_sub[fi]))
        max_nroots = 2*len(self.a_idxs) + 1
        charges = np.zeros ((max_nroots, self.nfrags), dtype=np.int32)
        spins = np.asarray ([[n[0]-n[1] for n in self.nelecas_sub] for i in range(max_nroots)]) 
        smults = np.abs (spins)+1 
        wfnsyms = wfnsyms = np.zeros ((max_nroots, self.nfrags), dtype=np.int32)

        lsi_nelecas_sub = [self.nelecas_sub]  # original state is the first one
        self.nroots = 1
        for a_idx, i_idx in zip(self.a_idxs, self.i_idxs):
            Aci, nelecas_sub = self.getAci(a_idx, i_idx)
            A_dagger_ci, nelecas_sub_dagger = self.getAci(i_idx, a_idx)

            for ci, nelecas_sub in zip([Aci, A_dagger_ci], [nelecas_sub, nelecas_sub_dagger]):
                if ci is not None:
                    for fi, ci_f in enumerate(ci):
                        self.ci[fi].append(ci_f)
                        charges[self.nroots, fi] = self.ncas_sub[fi] - sum(nelecas_sub[fi])
                        spins[self.nroots, fi] = nelecas_sub[fi][0] - nelecas_sub[fi][1]
                        smults[self.nroots, fi] = abs(spins[self.nroots, fi]) + 1    
                    lsi_nelecas_sub.append(nelecas_sub)
                    self.nroots += 1

        print("nroots prepared:", self.nroots)
        charges = charges[:self.nroots]
        spins = spins[:self.nroots]
        smults = smults[:self.nroots]
        wfnsyms = wfnsyms[:self.nroots]
        self.weights = np.zeros(self.nroots)
        self.weights[0] = 1.0  # original state has weight 1,
        



        # self.nelecas_sub = [[np.array(x) for x in nelecas_sub_buffer[fi]] for fi in range(self.nfrags)]

        # for fi in range(self.nfrags):
        #     self.fciboxes[fi] = get_h1e_zipped_fcisolver(state_average_n_mix (self._las, fciboxes_list[fi], [1.0] + [0.0] * (len(fciboxes_list[fi]) - 1)).fcisolver)
            

        self.fciboxes = [get_h1e_zipped_fcisolver (state_average_n_mix (
            self._las, [csf_solver (self._las.mol, smult=s2p1).set (charge=c, spin=m2, wfnsym=ir)
              for c, m2, s2p1, ir in zip (c_r, m2_r, s2p1_r, ir_r)], self.weights).fcisolver)
                for c_r, m2_r, s2p1_r, ir_r in zip (charges.T, spins.T, smults.T, wfnsyms.T)]    
                # self.e_states 
        # self.e_states = self.get_e_states()   # yeah indeed this is not necessary

    def kernel (self, **kwargs):
        self.prepare_states_()
        return LASSI.kernel (self, **kwargs)

    def filter_spaces (self, las):
        # Hook for child methods
        return las

    # make_lroots = make_lroots
    # prepare_states = prepare_states
    # as_scanner=as_scanner

if __name__ == "__main__":
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

    epsilon = 0.001
    all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(las, epsilon)
    # sort the selected excitations by gradiDent magnitude
    gredients = np.array(g_sel)[:,0]
    sorted_indices = np.argsort(-np.abs(gredients))
    a_idxs_selected = [a_idxs_selected[i] for i in sorted_indices] 
    i_idxs_selected = [i_idxs_selected[i] for i in sorted_indices]


    lsi_lscc = LASSI_LSCC(las, a_idxs_selected, i_idxs_selected, frag_orbs=frag_atom_list)
    e_roots, si_rq = lsi_lscc.kernel()
    print ("LASSI-LSCC energy =", e_roots[0])
    # print("ground state si vector =", si_rq[:,0])
