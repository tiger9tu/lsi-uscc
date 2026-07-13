from mrh.my_pyscf.lassi import LASSI
from mrh.my_pyscf.mcscf.addons import state_average_n_mix, get_h1e_zipped_fcisolver
from mrh.my_pyscf.mcscf.productstate import ImpureProductStateFCISolver
from mrh.my_pyscf.lassi.op_o1.frag import FragTDMInt
import numpy as np
from pyscf import lib
from pyscf.scf.addons import canonical_orth_ as _pyscf_canonical_orth
from pyscf.fci import addons
from pyscf.fci.direct_spin1 import trans_rdm12s as _fci_tdm12s
from pyscf.fci import cistring as _fci_cistring
from pyscf.csf_fci import csf_solver
from helper.op_ci import apply_operator_string_fci
from helper import util
from copy import deepcopy


# ── Optimized FragTDMInt subclass ──────────────────────────────────────────────

class LSIFragTDMInt(FragTDMInt):
    """FragTDMInt subclass that reuses precomputed spectator TDMs for within-group pairs.

    For two excited states A|LAS_{r,j1}> and A|LAS_{r,j2}> from the same (A, r)
    group and a spectator fragment fi, the transition density matrix between them
    equals the reference intra-rootspace TDM dm1[r][r][k1, k2].  Instead of
    recomputing these with trans_rdm12s, we look them up from a precomputed block
    built once per (fi, r_pool) pair.

    The cache is stored on the LAS object as ``las._lsi_tdm_cache`` and has the
    structure::

        {
          'spectator_skip': {fi: [(rs_a, rs_b), ...]},
          'pair_tdms': {(rs_a, rs_b): {fi: (dm1_1x1, dm2_1x1, ovlp_1x1)}},
        }
    """

    def __init__(self, las, ci, hopping_index, zerop_index, onep_index, norb, nroots, nelec_rs,
                 rootaddr, fragaddr, idx_frag, mask_ints, **kwargs):
        cache = getattr(las, '_lsi_tdm_cache', None)
        if cache is not None:
            skip = cache.get('spectator_skip', {}).get(idx_frag, [])
            if skip:
                mask_ints = mask_ints.copy()
                for rs_a, rs_b in skip:
                    mask_ints[rs_a, rs_b] = False
                    mask_ints[rs_b, rs_a] = False
        super().__init__(las, ci, hopping_index, zerop_index, onep_index, norb, nroots, nelec_rs,
                         rootaddr, fragaddr, idx_frag, mask_ints, **kwargs)
        if cache is not None:
            self._fill_from_cache(cache)

    def _fill_from_cache(self, cache):
        fi = self.idx_frag
        for (rs_a, rs_b), frag_data in cache.get('pair_tdms', {}).items():
            if fi not in frag_data:
                continue
            dm1_val, dm2_val, ovlp_val = frag_data[fi]
            ir = self.unique_root[rs_a]
            jr = self.unique_root[rs_b]
            if ir < jr:
                ir, jr = jr, ir
            if ir == jr:
                continue  # degenerate unique roots; skip to avoid corruption
            if self.dm1[ir][jr] is None:
                self.dm1[ir][jr] = np.ascontiguousarray(dm1_val)
            if dm2_val is not None and self.dm2[ir][jr] is None:
                self.dm2[ir][jr] = np.ascontiguousarray(dm2_val)
            if self.ovlp[ir][jr] is None:
                self.ovlp[ir][jr] = np.ascontiguousarray(ovlp_val)
                self.ovlp[jr][ir] = np.ascontiguousarray(ovlp_val.conj().T)


# ── Main solver class ─────────────────────────────────────────────────────────

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
        smult_si   : target total spin multiplicity for the SI diagonalization.
                     If set, Davidson diagonalization is used by default because
                     the in-core path does not enforce the target spin sector.
    """

    def __init__(self, las_or_lsi, a_idxs, i_idxs,
                 state=0, threshold=0.01, top_m=None, lindep_thresh=None, norm_thresh=None,
                 opt=1, **kwargs):
        self.a_idxs = a_idxs
        self.i_idxs = i_idxs
        self._smult_si = smult_si
        self._n_ref_rs = None
        self._exc_rs_meta = []
        self._lsi_tdm_cache = None
        self._fragint_class = None
        self._norm_thresh = norm_thresh if norm_thresh is not None else 1e-12

        if isinstance(las_or_lsi, LASSI):
            # lsi-luscc: LASSI object provided
            self._ref_lsi = las_or_lsi
            self._lsi_state = state
            self._si_threshold = threshold
            self._top_m = top_m  # if set, select top-m by |ci| regardless of threshold
            las = las_or_lsi._las
            # lsi-luscc generates many near-degenerate states; tighter lindep
            # threshold is needed to avoid a near-singular orthogonal basis.
            self._lindep_thresh = lindep_thresh if lindep_thresh is not None else 1e-3
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
            ci_f_new = ci_f_new / np.linalg.norm(ci_f_new.ravel())
            Aci[fi] = ci_f_new
            nelecas_sub[fi] = (neleca, nelecb)

        return Aci, nelecas_sub

    def _get_active_frags(self, a_idx, i_idx):
        """Return frozenset of fragment indices touched by operator (a_idx, i_idx)."""
        frag_orbs_start = [0]
        for norb_f in self.ncas_sub[:-1]:
            frag_orbs_start.append(frag_orbs_start[-1] + norb_f)
        active = set()
        for idx in list(a_idx) + list(i_idx):
            spatial = idx % self.ncas
            fi = int(np.searchsorted(frag_orbs_start, spatial, side='right') - 1)
            active.add(fi)
        return frozenset(active)

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
            fragaddr = None  # not used for las-luscc (no within-group pairs)

            def _ref_ci_and_nelec(j):
                ref_ci = [original_ci[fi][j] for fi in range(self.nfrags)]
                ref_nelec = [tuple(nelec_frs[fi, j]) for fi in range(self.nfrags)]
                return ref_ci, ref_nelec, j

        # ── Group reference states by rootspace r ─────────────────────────
        # Multiple sig_indices j can map to the same rootspace r when the
        # source LASSI object has lroots > 1 for that rootspace.  Using the
        # full ref.ci[fi][r] array (already LASCI-orthogonalised per frag)
        # as a single multi-root entry lets op_o1 vectorise over lroots pairs
        # without creating spurious Cartesian-product states, because LASCI
        # guarantees that per-fragment CI vectors within a rootspace are
        # independent bases — the Cartesian product IS the correct product
        # state expansion for that rootspace.
        ref_rs_seen = {}   # r -> pool_idx of first occurrence
        new_ci = [[] for _ in range(self.nfrags)]
        self.nroots = 0

        for j in sig_indices:
            _, _, r = _ref_ci_and_nelec(j)
            if r in ref_rs_seen:
                continue  # already added this rootspace's full CI
            ref_rs_seen[r] = self.nroots
            for fi in range(self.nfrags):
                new_ci[fi].append(ref.ci[fi][r] if self._ref_lsi is not None
                                  else original_ci[fi][r])
            charges[self.nroots] = _charges[r]
            spins  [self.nroots] = _spins  [r]
            smults [self.nroots] = _smults [r]
            wfnsyms[self.nroots] = _wfnsyms[r]
            self.nroots += 1

        n_ref_rs = self.nroots
        self._n_ref_rs = n_ref_rs
        self._exc_rs_meta = []

        # ── Excited states: one rootspace per valid (A, j) combination ────
        # A|LAS_j⟩ states from different reference product states are
        # *correlated*: the per-fragment CI vectors from different j's cannot
        # be stacked into a multi-root array without creating spurious
        # Cartesian-product states (because the fragment CIs come from
        # different j's and are not jointly LASCI-orthogonalised).
        # We therefore keep one rootspace per excited state, same as before.
        for a_idx, i_idx in zip(self.a_idxs, self.i_idxs):
            active_frags = self._get_active_frags(a_idx, i_idx)
            for j in sig_indices:
                ref_ci_j, ref_nelecas_sub_j, r_source = _ref_ci_and_nelec(j)
                r_pool = ref_rs_seen.get(r_source)

                for dir_idx, (a, i) in enumerate([(a_idx, i_idx), (i_idx, a_idx)]):
                    Aci, nelecas_sub_new = self.getAci(
                        a, i, ref_ci=ref_ci_j, ref_nelecas_sub=ref_nelecas_sub_j)
                    if Aci is not None:
                        # Track metadata for the within-group TDM optimization
                        if fragaddr is not None and r_pool is not None:
                            ki_per_frag = [int(fragaddr[fi, j]) for fi in range(self.nfrags)]
                        else:
                            ki_per_frag = [0] * self.nfrags
                        self._exc_rs_meta.append({
                            'group_key': (tuple(a_idx), tuple(i_idx), r_pool, dir_idx),
                            'active_frags': active_frags,
                            'ref_rs_pool': r_pool,
                            'ki_per_frag': ki_per_frag,
                            'nelecas': [tuple(nelecas_sub_new[fi]) for fi in range(self.nfrags)],
                        })

                        for fi, ci_f in enumerate(Aci):
                            new_ci[fi].append(ci_f)
                            charges[self.nroots, fi] = (
                                self.ncas_sub[fi] - sum(nelecas_sub_new[fi]))
                            spins  [self.nroots, fi] = (
                                nelecas_sub_new[fi][0] - nelecas_sub_new[fi][1])
                            smults [self.nroots, fi] = (
                                abs(spins[self.nroots, fi]) + 1)
                        self.nroots += 1

        self.e_states_meaningless = True

        from mrh.my_pyscf.lassi.citools import get_lroots as _get_lroots
        lroots_new = _get_lroots(new_ci)
        n_total = int(np.sum(np.prod(lroots_new, axis=0)))
        n_exc_states = n_total - int(np.sum(np.prod(lroots_new[:, :n_ref_rs], axis=0)))
        lib.logger.info(self,
            'nroots prepared: %d states (%d reference + %d excited) '
            'in %d rootspaces (%d ref + %d exc; avg %.1f states/rootspace)',
            n_total, n_total - n_exc_states, n_exc_states,
            self.nroots, n_ref_rs, self.nroots - n_ref_rs,
            n_total / max(self.nroots, 1))

        charges = charges[:self.nroots]
        spins   = spins  [:self.nroots]
        smults  = smults [:self.nroots]
        wfnsyms = wfnsyms[:self.nroots]

        self.ci = new_ci
        self.weights = np.zeros(self.nroots)
        self.weights[:n_ref_rs] = 1.0

        self.fciboxes = [get_h1e_zipped_fcisolver(state_average_n_mix(
            self._las, [csf_solver(self._las.mol, smult=s2p1).set(
                charge=c, spin=m2, wfnsym=ir)
                for c, m2, s2p1, ir in zip(c_r, m2_r, s2p1_r, ir_r)],
            self.weights).fcisolver)
            for c_r, m2_r, s2p1_r, ir_r in zip(
                charges.T, spins.T, smults.T, wfnsyms.T)]

    # ------------------------------------------------------------------
    # Within-group spectator TDM precomputation (Optimization 2)
    # ------------------------------------------------------------------

    def _compute_spectator_block(self, fi, ref_rs_pool, nelec_fi):
        """Compute the full M_r × M_r TDM block for spectator fragment fi.

        Args:
            fi          : fragment index
            ref_rs_pool : LSI-LUSCC rootspace index for the source reference rootspace
            nelec_fi    : (neleca, nelecb) electron count for fragment fi

        Returns:
            (dm1_block, dm2_block, ovlp_block) with shapes
            (M_r, M_r, 2, norb, norb), (M_r, M_r, 4, norb, norb, norb, norb), (M_r, M_r)
        """
        ci_block = self.ci[fi][ref_rs_pool]          # 3D (M_r, na, nb) or 2D (na, nb)
        if ci_block.ndim == 2:
            ci_block = ci_block[None, :]             # → (1, na, nb)
        M_r = ci_block.shape[0]
        norb_fi = self.ncas_sub[fi]
        na, nb = ci_block.shape[1], ci_block.shape[2]

        dm1_block  = np.zeros((M_r, M_r, 2, norb_fi, norb_fi))
        dm2_block  = np.zeros((M_r, M_r, 4, norb_fi, norb_fi, norb_fi, norb_fi))
        ovlp_block = np.zeros((M_r, M_r))

        for k_a in range(M_r):
            bra = ci_block[k_a].reshape(na, nb)
            for k_b in range(M_r):
                ket = ci_block[k_b].reshape(na, nb)
                ovlp_block[k_a, k_b] = np.dot(bra.ravel().conj(), ket.ravel())
                d1s, d2s = _fci_tdm12s(bra, ket, norb_fi, nelec_fi)
                dm1_block[k_a, k_b] = np.stack(d1s, axis=0).transpose(0, 2, 1)
                dm2_block[k_a, k_b] = np.stack(d2s, axis=0)

        return dm1_block, dm2_block, ovlp_block

    def _build_lsi_tdm_cache(self):
        """Precompute spectator TDMs for within-group pairs (Optimization 2).

        Within each (operator, reference-rootspace, direction) group of excited
        states, spectator-fragment TDMs are identical to reference intra-rootspace
        TDMs already available from the reference CI arrays.  This method computes
        those blocks once per (fragment, reference-rootspace) pair and caches the
        per-pair slices so LSIFragTDMInt can skip the corresponding trans_rdm12s
        calls during _init_crunch_.

        Returns None if there are no within-group pairs to optimise (e.g. all
        groups have only one member).
        """
        if not self._exc_rs_meta:
            return None

        n_ref_rs = self._n_ref_rs

        # Group excited rootspaces by group_key
        groups = {}
        for exc_idx, meta in enumerate(self._exc_rs_meta):
            gk = meta['group_key']
            rs_exc = n_ref_rs + exc_idx
            if gk not in groups:
                groups[gk] = {
                    'rs_list': [],
                    'ki_per_frag_list': [],
                    'nelecas_list': [],
                    'active_frags': meta['active_frags'],
                    'ref_rs_pool': meta['ref_rs_pool'],
                }
            groups[gk]['rs_list'].append(rs_exc)
            groups[gk]['ki_per_frag_list'].append(meta['ki_per_frag'])
            groups[gk]['nelecas_list'].append(meta['nelecas'])

        spectator_skip = {}   # fi → [(rs_a, rs_b), ...]
        pair_tdms = {}        # (rs_a, rs_b) → {fi: (dm1_1x1, dm2_1x1, ovlp_1x1)}
        spec_block_cache = {} # (fi, r_pool) → (dm1_block, dm2_block, ovlp_block)
        n_pairs_saved = 0

        for g in groups.values():
            rs_list = g['rs_list']
            if len(rs_list) < 2:
                continue

            active_frags   = g['active_frags']
            ref_rs_pool    = g['ref_rs_pool']
            ki_list        = g['ki_per_frag_list']   # ki_list[state_idx][fi]
            nelecas_list   = g['nelecas_list']        # nelecas_list[state_idx][fi]

            if ref_rs_pool is None:
                continue  # safety: no source rootspace recorded

            for idx_a in range(len(rs_list)):
                for idx_b in range(idx_a):
                    rs_a = rs_list[idx_a]   # rs_a > rs_b (added in order)
                    rs_b = rs_list[idx_b]

                    frag_data = {}
                    for fi in range(self.nfrags):
                        if fi in active_frags:
                            continue  # let parent compute active-fragment TDMs

                        # Ensure the spectator block for (fi, ref_rs_pool) is built
                        cache_key = (fi, ref_rs_pool)
                        if cache_key not in spec_block_cache:
                            nelec_fi = nelecas_list[idx_a][fi]  # same for all states (spectator)
                            spec_block_cache[cache_key] = self._compute_spectator_block(
                                fi, ref_rs_pool, nelec_fi)

                        dm1_blk, dm2_blk, ovlp_blk = spec_block_cache[cache_key]
                        k_a = ki_list[idx_a][fi]
                        k_b = ki_list[idx_b][fi]

                        # Slice out the (1,1,...) sub-block for this specific pair
                        dm1_val  = dm1_blk [k_a:k_a+1, k_b:k_b+1]
                        dm2_val  = dm2_blk [k_a:k_a+1, k_b:k_b+1]
                        ovlp_val = ovlp_blk[k_a:k_a+1, k_b:k_b+1]

                        frag_data[fi] = (dm1_val, dm2_val, ovlp_val)
                        spectator_skip.setdefault(fi, []).append((rs_a, rs_b))

                    if frag_data:
                        pair_tdms[(rs_a, rs_b)] = frag_data
                        n_pairs_saved += len(frag_data)

        if not pair_tdms:
            return None

        n_spec_blocks = len(spec_block_cache)
        lib.logger.info(self,
            'LSI-LUSCC TDM cache: %d within-group pairs optimised across %d spectator '
            'fragment-rootspace blocks (%.0f trans_rdm12s calls avoided, '
            '%d blocks computed once)',
            len(pair_tdms), n_spec_blocks, n_pairs_saved, n_spec_blocks)

        return {'spectator_skip': spectator_skip, 'pair_tdms': pair_tdms}

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    def _filter_smult_roots_(self, smult_si, tol=1e-4):
        target_s = (smult_si - 1) / 2
        target_s2 = target_s * (target_s + 1)
        s2 = np.asarray(self.si.s2)
        idx = np.where(np.abs(s2 - target_s2) <= tol)[0]
        if len(idx) == 0:
            raise RuntimeError(
                f"LSI_LUSCC found no roots with spin multiplicity {smult_si} "
                f"(target <S^2>={target_s2})")

        self.e_roots = self.e_roots[idx]
        si = self.si[:, idx]
        self.s2 = np.asarray(self.si.s2)[idx]
        self.nelec = [self.si.nelec[i] for i in idx]
        self.wfnsym = [self.si.wfnsym[i] for i in idx]
        self.rootsym = np.asarray(self.si.rootsym)[idx]
        self.si = tag_array(
            si, s2=self.s2, nelec=self.nelec, wfnsym=self.wfnsym,
            rootsym=self.rootsym, break_symmetry=self.si.break_symmetry,
            soc=self.si.soc)
        return self.e_roots, self.si

    def kernel(self, **kwargs):
        requested_smult = self._smult_si if self._smult_si is not None else kwargs.get('smult_si')
        injected_davidson = False
        if self._smult_si is not None:
            kwargs.setdefault('smult_si', self._smult_si)
        if requested_smult is not None and 'davidson_only' not in kwargs:
            kwargs['davidson_only'] = True
            injected_davidson = True

        self.prepare_states_()

        # Build spectator TDM cache and activate the optimised FragTDMInt subclass
        cache = self._build_lsi_tdm_cache()
        if cache is not None:
            self._lsi_tdm_cache = cache
            self._fragint_class = LSIFragTDMInt

        import mrh.my_pyscf.lassi.citools as _citools
        import mrh.my_pyscf.lassi.basis as _basis
        import mrh.my_pyscf.lassi.sisolver as _sisolver
        import mrh.my_pyscf.lassi.spaces as _spaces

        def _safe_canonical_orth(ovlp, thr=1e-7):
            ovlp = np.asarray(ovlp)
            diag = np.real(np.diag(ovlp))
            keep = np.isfinite(diag) & (diag > self._norm_thresh)
            if np.all(keep):
                return _pyscf_canonical_orth(ovlp, thr=thr)

            if np.count_nonzero(keep):
                x_keep = _pyscf_canonical_orth(ovlp[np.ix_(keep, keep)], thr=thr)
                xmat = np.zeros((ovlp.shape[0], x_keep.shape[1]), dtype=x_keep.dtype)
                xmat[keep] = x_keep
            else:
                xmat = np.zeros((ovlp.shape[0], 0), dtype=ovlp.dtype)
            lib.logger.warn(
                self,
                'Dropped %d low-norm LASSI model states before canonical orthogonalization '
                '(norm_thresh=%g)',
                ovlp.shape[0] - np.count_nonzero(keep), self._norm_thresh)
            return xmat

        _thresh_modules = [_citools, _basis, _sisolver, _spaces]
        _old_thresh = {}
        for _mod in _thresh_modules:
            if hasattr(_mod, "LINDEP_THRESH"):
                _old_thresh[_mod] = _mod.LINDEP_THRESH
                _mod.LINDEP_THRESH = self._lindep_thresh
        _old_basis_canonical_orth = _basis.canonical_orth_
        _old_sisolver_canonical_orth = _sisolver.canonical_orth_
        _basis.canonical_orth_ = _safe_canonical_orth
        _sisolver.canonical_orth_ = _safe_canonical_orth
        try:
            try:
                result = LASSI.kernel(self, **kwargs)
            except AssertionError:
                if requested_smult is None or not injected_davidson:
                    raise
                lib.logger.warn(
                    self,
                    "Spin-coupled Davidson diagonalization is not available for "
                    "this LAS-LUSCC state space; falling back to direct "
                    "diagonalization followed by <S^2> root filtering.")
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop('smult_si', None)
                fallback_kwargs['davidson_only'] = False
                result = LASSI.kernel(self, **fallback_kwargs)
                result = self._filter_smult_roots_(requested_smult)
        finally:
            for _mod, _thr in _old_thresh.items():
                _mod.LINDEP_THRESH = _thr
            _basis.canonical_orth_ = _old_basis_canonical_orth
            _sisolver.canonical_orth_ = _old_sisolver_canonical_orth
            self._lsi_tdm_cache = None
            self._fragint_class = None
        return result

    def filter_spaces(self, las):
        return las
