from mrh.my_pyscf.lassi import LASSI
from mrh.my_pyscf.mcscf.addons import state_average_n_mix, get_h1e_zipped_fcisolver
from mrh.my_pyscf.mcscf.productstate import ImpureProductStateFCISolver
import numpy as np
from pyscf import lib
from pyscf.fci import addons
from pyscf.csf_fci import csf_solver
from helper.op_ci import apply_operator_string_fci
from helper import util
from copy import deepcopy


class LSI_LUSCC(LASSI):
    """Perform LUSCC using the LASSI framework.

    Supports two use cases (unified):

    las-luscc (special case):
        LSI_LUSCC(las, a_idxs, i_idxs)
        Single LAS reference state.  Excitations Aᵢ are applied to |las⟩.

    lsi-luscc (general case):
        LSI_LUSCC(lsi, a_idxs, i_idxs, state=0, threshold=0.01)
        LASSI reference.  |lsi⟩ = Σⱼ cⱼ|las_j⟩.  Significant components
        (|cⱼ| > threshold) are used as reference states; each excitation Aᵢ is
        applied to every significant component.

    Args:
        las_or_lsi : LAS object *or* post-kernel LASSI object
        a_idxs     : list of creation-operator index tuples (pre-selected)
        i_idxs     : list of annihilation-operator index tuples (pre-selected)
        state      : which LASSI eigenstate to read SI coefficients from (default 0)
        threshold  : |SI coefficient| cutoff for selecting significant LAS
                     components (default 0.01)
    """

    def __init__(self, las_or_lsi, a_idxs, i_idxs,
                 state=0, threshold=0.01, top_m=None, lindep_thresh=None, opt=1, **kwargs):
        self.a_idxs = a_idxs
        self.i_idxs = i_idxs

        if isinstance(las_or_lsi, LASSI):
            # lsi-luscc: LASSI object provided
            self._ref_lsi = las_or_lsi
            self._lsi_state = state
            self._si_threshold = threshold
            self._top_m = top_m  # if set, select top-m by |ci| regardless of threshold
            las = las_or_lsi._las
            # lsi-luscc generates many near-degenerate states; tighter lindep
            # threshold is needed to avoid a near-singular orthogonal basis.
            self._lindep_thresh = lindep_thresh if lindep_thresh is not None else 1e-4
        else:
            # las-luscc: bare LAS object (single-root special case)
            self._ref_lsi = None
            self._top_m = None
            self._lindep_thresh = lindep_thresh if lindep_thresh is not None else 1e-5
            las = las_or_lsi

        LASSI.__init__(self, las, opt=opt, **kwargs)

    # ------------------------------------------------------------------
    # Significant component selection
    # ------------------------------------------------------------------

    def _select_sig_indices(self):
        """Return indices of significant LAS components for the reference state."""
        if self._ref_lsi is not None and self._ref_lsi.si is not None:
            si_vec = self._ref_lsi.si[:, self._lsi_state]
            if self._top_m is not None:
                # Count-based: select top-m by |ci|
                m = min(self._top_m, len(si_vec))
                sig_indices = np.argsort(-np.abs(si_vec))[:m].tolist()
            else:
                sig_indices = [j for j in range(len(si_vec))
                               if abs(si_vec[j]) > self._si_threshold]
                if len(sig_indices) == 0:
                    # Fall back to dominant component
                    sig_indices = [int(np.argmax(np.abs(si_vec)))]
        else:
            # Single-root case: only root 0
            sig_indices = [0]
        return sig_indices

    # ------------------------------------------------------------------
    # Operator application
    # ------------------------------------------------------------------

    def getAci(self, a_idx, i_idx, ref_ci=None, ref_nelecas_sub=None):
        """Apply excitation operator to a reference CI vector.

        A|ψ⟩ = a₀a₁…i₁i₀|ψ⟩

        Args:
            a_idx            : creation operator indices (spinless)
            i_idx            : annihilation operator indices (spinless)
            ref_ci           : list of per-fragment CI vectors to act on.
                               Defaults to root 0 of self.ci (original behaviour).
            ref_nelecas_sub  : list of (neleca, nelecb) per fragment for ref_ci.
                               Defaults to self.nelecas_sub.

        Returns:
            (Aci, nelecas_sub) or (None, None) if the excitation is invalid.
        """
        frag_orbs_start = [0]
        for norb_f in self.ncas_sub[:-1]:
            frag_orbs_start.append(frag_orbs_start[-1] + norb_f)

        if ref_ci is None:
            ref_ci = [frag_ci[0] for frag_ci in self.ci]
        if ref_nelecas_sub is None:
            ref_nelecas_sub = self.nelecas_sub

        Aci = deepcopy(ref_ci)
        frag_ops = [[] for _ in range(self.nfrags)]

        for op_type, idx_list in [('ann', i_idx), ('cre', a_idx[::-1])]:
            for idx in idx_list:
                spatial = idx % self.ncas
                spin = idx // self.ncas
                frag_idx = np.searchsorted(frag_orbs_start, spatial, side='right') - 1
                idx_in_frag = spatial - frag_orbs_start[frag_idx]
                frag_ops[frag_idx].append((op_type, idx_in_frag, spin))

        nelecas_sub = deepcopy(list(ref_nelecas_sub))
        for fi, ci_f in enumerate(Aci):
            if len(frag_ops[fi]) == 0:
                continue
            ci_f_new, (neleca, nelecb) = apply_operator_string_fci(
                ci_f, self.ncas_sub[fi], ref_nelecas_sub[fi], frag_ops[fi])
            if ci_f_new is None or np.all(ci_f_new == 0):
                return None, None
            Aci[fi] = ci_f_new
            nelecas_sub[fi] = (neleca, nelecb)

        return Aci, nelecas_sub

    # ------------------------------------------------------------------
    # State preparation
    # ------------------------------------------------------------------

    def prepare_states_(self):
        from mrh.my_pyscf.mcscf.lasci import get_space_info
        from mrh.my_pyscf.lassi.citools import get_lroots, get_rootaddr_fragaddr

        sig_indices = self._select_sig_indices()
        n_sig = len(sig_indices)
        max_nroots = n_sig * (1 + 2 * len(self.a_idxs))

        charges = np.zeros((max_nroots, self.nfrags), dtype=np.int32)
        spins   = np.zeros((max_nroots, self.nfrags), dtype=np.int32)
        smults  = np.ones ((max_nroots, self.nfrags), dtype=np.int32)
        wfnsyms = np.zeros((max_nroots, self.nfrags), dtype=np.int32)

        if self._ref_lsi is not None:
            # lsi-luscc: sig_indices are product state indices (0..nprod-1).
            # Use get_rootaddr_fragaddr to map each product state j to its
            # rootspace r and per-fragment CI index ki.
            ref = self._ref_lsi
            _charges, _spins, _smults, _wfnsyms = get_space_info(ref)
            nelec_frs = self.get_nelec_frs(ref)
            lroots = get_lroots(ref.ci)
            rootaddr, fragaddr = get_rootaddr_fragaddr(lroots)

            def _ref_ci_and_nelec(j):
                r = rootaddr[j]
                ref_ci = []
                for fi in range(self.nfrags):
                    ki = fragaddr[fi, j]
                    ci_fir = ref.ci[fi][r]
                    ref_ci.append(ci_fir[ki] if ci_fir.ndim > 2 else ci_fir)
                ref_nelec = [tuple(nelec_frs[fi, r]) for fi in range(self.nfrags)]
                return ref_ci, ref_nelec, r
        else:
            # las-luscc: sig_indices are simple LAS root indices.
            _charges, _spins, _smults, _wfnsyms = get_space_info(self._las)
            nelec_frs = self.get_nelec_frs(self._las)
            original_ci = self.ci

            def _ref_ci_and_nelec(j):
                ref_ci = [original_ci[fi][j] for fi in range(self.nfrags)]
                ref_nelec = [tuple(nelec_frs[fi, j]) for fi in range(self.nfrags)]
                return ref_ci, ref_nelec, j

        # Pool starts with the significant reference states
        new_ci = [[] for _ in range(self.nfrags)]
        for pool_idx, j in enumerate(sig_indices):
            ref_ci_j, _, r = _ref_ci_and_nelec(j)
            for fi in range(self.nfrags):
                new_ci[fi].append(ref_ci_j[fi])
            charges[pool_idx] = _charges[r]
            spins  [pool_idx] = _spins  [r]
            smults [pool_idx] = _smults [r]
            wfnsyms[pool_idx] = _wfnsyms[r]

        self.nroots = n_sig

        # Apply each selected excitation to every significant reference state
        for a_idx, i_idx in zip(self.a_idxs, self.i_idxs):
            for j in sig_indices:
                ref_ci_j, ref_nelecas_sub_j, _ = _ref_ci_and_nelec(j)

                for a, i in [(a_idx, i_idx), (i_idx, a_idx)]:
                    Aci, nelecas_sub_new = self.getAci(
                        a, i, ref_ci=ref_ci_j, ref_nelecas_sub=ref_nelecas_sub_j)
                    if Aci is not None:
                        for fi, ci_f in enumerate(Aci):
                            new_ci[fi].append(ci_f)
                            charges[self.nroots, fi] = (
                                self.ncas_sub[fi] - sum(nelecas_sub_new[fi]))
                            spins  [self.nroots, fi] = (
                                nelecas_sub_new[fi][0] - nelecas_sub_new[fi][1])
                            smults [self.nroots, fi] = (
                                abs(spins[self.nroots, fi]) + 1)
                        self.nroots += 1

        lib.logger.info(self, 'nroots prepared: %d (%d reference + %d excited)',
                        self.nroots, n_sig, self.nroots - n_sig)

        charges = charges[:self.nroots]
        spins   = spins  [:self.nroots]
        smults  = smults [:self.nroots]
        wfnsyms = wfnsyms[:self.nroots]

        self.ci = new_ci
        self.weights = np.zeros(self.nroots)
        self.weights[:n_sig] = 1.0  # equal weight to all reference states

        self.fciboxes = [get_h1e_zipped_fcisolver(state_average_n_mix(
            self._las, [csf_solver(self._las.mol, smult=s2p1).set(
                charge=c, spin=m2, wfnsym=ir)
                for c, m2, s2p1, ir in zip(c_r, m2_r, s2p1_r, ir_r)],
            self.weights).fcisolver)
            for c_r, m2_r, s2p1_r, ir_r in zip(
                charges.T, spins.T, smults.T, wfnsyms.T)]

    def kernel(self, **kwargs):
        self.prepare_states_()
        import mrh.my_pyscf.lassi.citools as _citools
        _old_thresh = _citools.LINDEP_THRESH
        _citools.LINDEP_THRESH = self._lindep_thresh
        try:
            result = LASSI.kernel(self, **kwargs)
        finally:
            _citools.LINDEP_THRESH = _old_thresh
        return result

    def filter_spaces(self, las):
        return las
