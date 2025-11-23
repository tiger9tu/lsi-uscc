# lasscf_test_framework_refactored.py
from __future__ import annotations
import numpy as np
from typing import Iterable, List, Any, Tuple, Dict, Optional, TypedDict
from dataclasses import dataclass, field
import functools, time, sys

# External deps
from pyscf import gto, scf, mcscf
from scipy.linalg import eigh

# mrh stack
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.citools import grad, fockspace

# ------------------ Utilities ------------------
def timeit(func=None, *, label=None, logger=print):
    if func is None:
        return lambda f: timeit(f, label=label, logger=logger)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0] if args else None
        verbose = getattr(self, "VERBOSE", 1) if self is not None else 1
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        t1 = time.perf_counter()
        if verbose >= 1:
            logger(f"{label or func.__name__} took {(t1 - t0)*1000:.3f} ms")
        return out
    return wrapper

def print_matrix(mat):
    for row in mat:
        print("  ".join(f"{x:.17f}" for x in row))

def cilas2f(lasci, norb_f, nelec_f):
    """Convert LAS CI (per-fragment) to full Fock-space CI."""
    ci_f = []
    for i, ci in enumerate(lasci):
        ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
    return ci_f


# ------------------ Config type for molecules ------------------
class MolConfig(TypedDict, total=False):
    name : str
    xyz: str
    basis: str
    ncas: List[int]
    nelecas: List[int]
    spinsub: List[int]
    frag_atom_list: Any
    output: str
    symmetry: Optional[str]
    charge: Optional[int]
    spin: Optional[int]
    HF: Optional[str]
    frag_spin_orb: Optional[Dict[int, Tuple[int, ...]]]
    adj: Optional[List[List[int]]]

# ===============================================================
# Base test class: holds ALL configuration (no TestConfig anymore)
# ===============================================================
@dataclass
class LASCCTest:
    # --- Required molecule config ---
    mol_conf: MolConfig

    # --- Global knobs (were globals; now per-instance) ---
    VERBOSE: int = 3                 # print level
    LINEARIZE: bool = True           # linearize USCC building for NOQI
    THRESHOLD_COND: float = 8e5      # condition-number cutoff in NOQI
    AMPLITUDE: float = 1.0           # single-excitation amplitude for NOQI trials

    # --- Optional runtime toggles ---
    run_noqi_after_vqe: bool = True  # whether to run NOQI/NOCI after each VQE selection

    # --- Internal storage (reused across sweeps) ---
    mol: Optional[gto.Mole] = field(init=False, default=None)
    mf: Optional[scf.hf.SCF] = field(init=False, default=None)
    las: Optional[LASSCF] = field(init=False, default=None)
    casci_ref: Optional[mcscf.CASCI] = field(init=False, default=None)

    # Results organized per selection label
    energies: Dict[str, Dict[str, float]] = field(init=False, default_factory=dict)  # label -> {RHF, LASSCF, CASCI, LAS-USCC-VQE, LAS-USCC-NOQI}
    selection_records: Dict[str, Dict[str, Any]] = field(init=False, default_factory=dict)  # label -> {a_idxs, i_idxs, notes...}

    # ---------------- Core stack (run once) ----------------
    @timeit(label="RHF/ROHF")
    def run_rhf(self):
        symm = self.mol_conf.get('symmetry')
        chg  = self.mol_conf.get('charge')
        spin = self.mol_conf.get('spin')
        self.mol = gto.M(
            atom=self.mol_conf['xyz'],
            basis=self.mol_conf['basis'],
            symmetry=symm,
            charge=chg,
            spin=spin,
            verbose=max(self.VERBOSE, 4),  # leave PySCF at 4 if you're chatty
            output=self.mol_conf.get('output', None),
        )
        if self.mol_conf.get('HF') == 'ROHF':
            self.mf = scf.ROHF(self.mol).run()
        else:
            self.mf = scf.RHF(self.mol).run()

    @timeit(label="LASSCF kernel")
    def run_lasscf(self):
        assert self.mf is not None and self.mol is not None
        ncas = self.mol_conf['ncas']
        nele = self.mol_conf['nelecas']
        spinsub = self.mol_conf['spinsub']
        frag_atoms = self.mol_conf['frag_atom_list']

        las = LASSCF(self.mf, ncas, nele, spin_sub=spinsub, ouput=None)
        mo_loc = las.localize_init_guess(frag_atoms, self.mf.mo_coeff)
        las.kernel(mo_loc)
        self.las = las
        if self.VERBOSE >= 2:
            print(f"LASSCF energy: {las.e_tot:.17f}")

    @timeit(label="CASCI (ref) kernel")
    def run_casci(self):
        assert self.mf is not None and self.las is not None
        nact = sum(self.mol_conf['ncas'])
        nelec = sum(self.mol_conf['nelecas'])
        ref = mcscf.CASCI(self.mf, nact, nelec)
        ref.mo_coeff = self.las.mo_coeff
        ref.kernel()
        self.casci_ref = ref
        if self.VERBOSE >= 2:
            print(f"CASCI energy: {ref.e_tot:.17f}")

    def run_scf_stack_once(self, label: str = "base"):
        """Run RHF->LASSCF->CASCI exactly once (store reference energies under 'label')."""
        self.run_rhf()
        self.run_lasscf()
        self.run_casci()
        self.energies[label] = {
            "RHF": float(self.mf.e_tot),
            "LASSCF": float(self.las.e_tot),
            "CASCI": float(self.casci_ref.e_tot),
        }

    # ---------------- Excitation selection ----------------
    def select_excitations(self, *, epsilon: Optional[float]=None, factor: Optional[float]=None) -> Tuple[List, List, Dict[str, Any]]:
        """
        Choose excitations by either numeric epsilon (magnitude cutoff) or percentile 'factor' in (0,1].
        Returns (a_idxs_selected, i_idxs_selected, meta)
        """
        assert self.las is not None
        if factor is not None:
            g_all, _, _, _ = grad.get_grad_exact(self.las, epsilon=0.0)
            sortg = np.sort(np.abs(g_all))
            n = len(sortg)
            k = int(np.floor(factor * n))
            k = min(max(k, 1), n)
            epsilon = sortg[-k]
            if self.VERBOSE >= 2:
                print(f"[selection] factor={factor:.4f} -> epsilon={epsilon:.6e}")

        if epsilon is None:
            epsilon = 0.0

        all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(self.las, epsilon)

        if self.VERBOSE >= 2:
            print("total excitation count:", len(all_g))
            print("selected excitation count:", len(g_sel))

        meta = dict(epsilon=epsilon, factor=factor, total=len(all_g), selected=len(g_sel))
        return a_idxs_selected, i_idxs_selected, meta

    # ---------------- VQE run for a given selection ----------------
    @timeit(label="LAS-USCC-VQE kernel")
    def run_vqe(self, a_idxs_selected, i_idxs_selected) -> Tuple[mcscf.CASCI, float]:
        assert self.mf is not None and self.las is not None and self.mol is not None
        nact = sum(self.mol_conf['ncas'])
        nelec = sum(self.mol_conf['nelecas'])
        mc_uscc = mcscf.CASCI(self.mf, nact, nelec)
        mc_uscc.mo_coeff = self.las.mo_coeff
        mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(self.mol, a_idxs_selected, i_idxs_selected)
        mc_uscc.fcisolver.norb_f = self.mol_conf['ncas']
        # You can subclass and override to change freezing behavior:
        mc_uscc.fcisolver.frozen = getattr(self, "FROZEN", None)

        ci0 = cilas2f(self.las.ci, self.mol_conf['ncas'], self.mol_conf['nelecas'])
        mc_uscc.kernel(ci0=ci0)
        if not mc_uscc.converged and self.VERBOSE >= 1:
            print("Warning: LAS-USCC-VQE kernel hasn't converged")
        return mc_uscc, float(mc_uscc.e_tot)

    # ---------------- NOQI/NOCI helpers ----------------
    def _get_Sij(self, psi_i, psi_j, h) -> complex:
        ucj, _ = psi_j.hc_x(psi_j.x, h)[1:3]
        uci, _ = psi_i.hc_x(psi_i.x, h)[1:3]
        return uci.ravel().conj().dot(ucj.ravel())

    def _get_Sij_Hij(self, psi_i, psi_j, h) -> Tuple[complex, complex]:
        ucj, hucj = psi_j.hc_x(psi_j.x, h)[1:3]
        uci, huci = psi_i.hc_x(psi_i.x, h)[1:3]
        uci = uci.ravel()
        return uci.conj().dot(ucj.ravel()), uci.conj().dot(hucj.ravel())

    def _psi_selection(self, trial_psi, h) -> List[int]:
        selected_indices: List[int] = []
        S_inc = np.zeros((0, 0), dtype=np.complex128)
        for idx, psi in enumerate(trial_psi):
            n_sel = len(selected_indices)
            S_new = np.zeros((n_sel + 1, n_sel + 1), dtype=np.complex128)
            if n_sel > 0:
                S_new[:n_sel, :n_sel] = S_inc
                for j, jidx in enumerate(selected_indices):
                    S_new[n_sel, j] = self._get_Sij(psi, trial_psi[jidx], h)
                    S_new[j, n_sel] = self._get_Sij(trial_psi[jidx], psi, h)
            S_new[n_sel, n_sel] = self._get_Sij(psi, psi, h)
            cond = np.linalg.cond(S_new)
            if cond < self.THRESHOLD_COND:
                selected_indices.append(idx)
                S_inc = S_new
            elif self.VERBOSE >= 4:
                print(f"Discarding CI {idx} due to linear dependence (cond={cond:.2e})")
        return selected_indices

    @timeit(label="LAS-USCC-NOQI (generalized eig)")
    def run_noqi(self, mc_uscc: mcscf.CASCI, a_idxs_selected, i_idxs_selected) -> float:
        """Build trial PSIs by toggling one amplitude; solve Hc=ESc. Uses LINEARIZE and AMPLITUDE."""
        assert self.mf is not None and self.las is not None

        h1eff, e_core = mc_uscc.get_h1eff(mc_uscc.mo_coeff)
        h2eff = mc_uscc.get_h2eff()
        h = [e_core, h1eff, h2eff]

        nc = len(a_idxs_selected) + 1

        # Build a fresh FCISolver context to generate psi objects
        mc_tmp = mcscf.CASCI(self.mf, sum(self.mol_conf['ncas']), sum(self.mol_conf['nelecas']))
        mc_tmp.mo_coeff = self.las.mo_coeff
        mc_tmp.fcisolver = lasuccsd.FCISolver_USCC(self.mol, a_idxs_selected, i_idxs_selected)
        mc_tmp.fcisolver.norb_f = self.mol_conf['ncas']
        mc_tmp.fcisolver.frozen = getattr(self, "FROZEN", None)
        fci = mc_tmp.fcisolver

        las_ci0_f = cilas2f(self.las.ci, self.mol_conf['ncas'], self.mol_conf['nelecas'])
        norb = sum(self.mol_conf['ncas'])
        nelec = sum(self.mol_conf['nelecas'])
        norb_f = self.mol_conf['ncas']

        psis = []
        min_ci_eng = +np.inf
        for i in range(nc):
            psi = getattr(fci, 'psi', fci.build_psi(las_ci0_f, norb, norb_f, nelec, frozen=getattr(self, "FROZEN", None)))
            psi.uop.linearize = bool(self.LINEARIZE)
            if i < nc - 1:
                psi.x[psi.nconstr + i] = float(self.AMPLITUDE)
            energy = psi.energy_tot(psi.x, h)
            min_ci_eng = min(min_ci_eng, float(energy))
            psis.append(psi)

        if self.VERBOSE >= 2:
            print(f"NOQI trial states (incl ref): {nc}; min single-state E: {min_ci_eng:.12f}")

        sel = self._psi_selection(psis, h)
        sel_psis = [psis[i] for i in sel]
        if self.VERBOSE >= 2:
            print(f"Selected {len(sel)} / {nc} states after cond filter")

        # Build S/H and solve
        S = np.zeros((len(sel_psis), len(sel_psis)), dtype=np.complex128)
        H = np.zeros_like(S)
        for i in range(len(sel_psis)):
            for j in range(len(sel_psis)):
                S[i, j], H[i, j] = self._get_Sij_Hij(sel_psis[i], sel_psis[j], h)

        if self.VERBOSE >= 3:
            print("Condition number of S:", np.linalg.cond(S))
        if self.VERBOSE >= 4:
            print("S:"); print_matrix(S)
            print("H:"); print_matrix(H)

        eigvals, _ = eigh(H, S)
        e_noqi = float(np.min(eigvals.real))
        print(f"LAS-USCC-NOQI energy: {e_noqi:.17f}")
        return e_noqi

    # ---------------- High-level sweeps ----------------
    def run_selection_sweep(
        self,
        *,
        label_prefix: str,
        epsilons: Optional[List[float]] = None,
        factors: Optional[List[float]] = None
    ):
        """
        Reuse RHF/LASSCF/CASCI (already run) and scan multiple selections.
        For each selection, run VQE (and NOQI if enabled).
        """
        assert self.las is not None and self.casci_ref is not None

        base_energies = self.energies.get("base", {})
        if not base_energies:
            # allow different naming, but keep a copy of ref numbers
            self.energies["base"] = {
                "RHF": float(self.mf.e_tot),
                "LASSCF": float(self.las.e_tot),
                "CASCI": float(self.casci_ref.e_tot),
            }

        tasks: List[Tuple[str, Dict[str, Optional[float]]]] = []
        if epsilons:
            for eps in epsilons:
                tasks.append((f"{label_prefix}_eps_{eps:g}", dict(epsilon=float(eps), factor=None)))
        if factors:
            for f in factors:
                tasks.append((f"{label_prefix}_pct_{int(100*f)}", dict(epsilon=None, factor=float(f))))

        
        for label, kw in tasks:
            print() # newline
            print("=== Energies for selection ===\n")   
            a_idxs, i_idxs, meta = self.select_excitations(**kw)
            self.selection_records[label] = {"a_idxs": a_idxs, "i_idxs": i_idxs, "meta": meta}
            
            mc_uscc, e_vqe = self.run_vqe(a_idxs, i_idxs)
            self.energies[label] = {
                "RHF": self.energies["base"]["RHF"],
                "LASSCF": self.energies["base"]["LASSCF"],
                "CASCI": self.energies["base"]["CASCI"],
                "LAS-USCC-VQE": e_vqe,
            }
            if self.run_noqi_after_vqe:
                e_noqi = self.run_noqi(mc_uscc, a_idxs, i_idxs)
                self.energies[label]["LAS-USCC-NOQI"] = e_noqi


        return self.energies

# ===============================================================
# Example subclass: override defaults for a particular campaign
# ===============================================================
@dataclass
class MyNOQIStudy(LASCCTest):
    # Example overrides (you can create many subclasses with different knobs)
    VERBOSE: int = 4
    LINEARIZE: bool = True
    THRESHOLD_COND: float = 8e5
    AMPLITUDE: float = 1.0
    run_noqi_after_vqe: bool = True
    # Optionally: freezing policy
    FROZEN: Optional[str] = "CI"   # set to None or "CI"

# ===============================================================
# Minimal runnable example (replace xyz with your file content)
# ===============================================================
if __name__ == "__main__":
    # CLI override for amplitude preserved
    if len(sys.argv) > 1:
        try:
            AMPL = float(sys.argv[1])
            print(f"Using AMPLITUDE argument: {AMPL}")
        except ValueError:
            AMPL = 1.0
            print("Invalid AMPLITUDE argument; using default 1.0")
    else:
        AMPL = 1.0
    
    # Example H4 (replace by reading your geom files)
    with open(data_dir + '/h4.xyz', 'r', encoding='utf-8') as f:
        h4xyz = f.read()
    
    h4_sto3g: MolConfig = {
        'name': 'H4_STO3G',
        'xyz': h4xyz,
        'basis': 'sto-3g',
        'ncas': [2, 2],
        'nelecas': [2, 2],
        'spinsub': [1, 1],
        'frag_atom_list': ((0,1),(2,3)),
        'output': 'h4_sto3g.out',
        'symmetry': None, 'charge': None, 'spin': None, 'HF': None,
    }

    job = MyNOQIStudy(h4_sto3g, AMPLITUDE=AMPL)
    # Run SCF->LASSCF->CASCI ONCE
    job.run_scf_stack_once(label="base")

    # Sweep multiple selections without re-running LAS/CAS
    # Example: two epsilons and two percentile factors
    results = job.run_selection_sweep(
        label_prefix="scan",
        epsilons=[0.01, 0.001],
        factors=[0.05, 0.10],
    )

    print("\n=== Summary energies ===")
    for label, rec in results.items():
        print(f"[{label}]")
        for k, v in rec.items():
            print(f"  {k:>16s}: {v:.12f}")
