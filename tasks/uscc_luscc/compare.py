"""Shared machinery for the USCC-vs-LUSCC FCI-basis comparison.

System-specific drivers (h4.py, chn_10_10.py, ...) construct (mol, mf, las) and
call `run_comparison(label, mol, mf, las, fracs, out_dir, title_prefix)`. This
module owns:

  - Fock-space construction of |LAS>, |LAS_USCC>, |LAS_LUSCC> in a single
    consistent basis using LASUCCTrialState.dp_ci.
  - Re-diagonalization of H and S in the LSI_LUSCC-generated rootspace basis
    (basis-independence check against LASSI's eigenvalue).
  - In-plane / out-of-plane decomposition of |LAS_USCC>.
  - Per-frac 2-panel figures (schematic + true-deviation zoom) and a sweep plot.
"""

import time
import numpy as np
from pathlib import Path
from pyscf import mcscf
from mrh.exploratory.citools import fockspace, grad
from mrh.exploratory.unitary_cc import lasuccsd

from lcc import LSI_LUSCC
from helper.util import get_sorted_excitations


def build_rootspace_fci_vectors(lsi, psi):
    """Return a list of Fock-space FCI vectors, one per LSI_LUSCC rootspace.

    Each rootspace contributes a single product state (lroots == 1 in this
    setup). Per-fragment Hilbert CIs `lsi.ci[fi][r]` are converted to Fock
    space using (neleca, nelecb) read from the fcibox's per-root solver and
    combined via psi.dp_ci so the sign convention matches |LAS> and |LAS_USCC>.
    """
    vecs = []
    for r in range(lsi.nroots):
        ci_f_fock = []
        for fi in range(lsi.nfrags):
            ci_fr = lsi.ci[fi][r]
            if ci_fr.ndim == 3:
                assert ci_fr.shape[0] == 1
                ci_fr = ci_fr[0]
            solver = lsi.fciboxes[fi].fcisolvers[r]
            no = lsi.ncas_sub[fi]
            ne_tot = no - solver.charge
            neleca = (ne_tot + solver.spin) // 2
            nelecb = (ne_tot - solver.spin) // 2
            ci_f_fock.append(np.squeeze(
                fockspace.hilbert2fock(ci_fr, no, (neleca, nelecb))))
        vecs.append(psi.dp_ci(ci_f_fock))
    return vecs


def solve_luscc_in_fock_basis(vecs, psi, h_eff, lindep_thresh=1e-8):
    """Build H and S in the Fock-space basis {vecs} and solve H c = E S c."""
    n = len(vecs)
    S = np.empty((n, n))
    H = np.empty((n, n))
    Hvecs = [psi.contract_h2(list(h_eff), v.copy()) for v in vecs]
    for r in range(n):
        vr = vecs[r].ravel()
        for s in range(n):
            S[r, s] = float(np.dot(vr, vecs[s].ravel()))
            H[r, s] = float(np.dot(vr, Hvecs[s].ravel()))
    S = 0.5 * (S + S.T)
    H = 0.5 * (H + H.T)
    w, U = np.linalg.eigh(S)
    keep = w > lindep_thresh
    Uk = U[:, keep]
    wk = w[keep]
    X = Uk / np.sqrt(wk)
    Hp = X.T @ H @ X
    ep, Cp = np.linalg.eigh(Hp)
    return ep, X @ Cp


def energy_of(psi, h_eff, fcivec):
    hfci = psi.contract_h2(list(h_eff), fcivec.copy())
    return float(np.dot(fcivec.ravel(), hfci.ravel()))


def spin_square_of(psi, fcivec):
    """<S^2> of a Fock-space FCI vector, using the lasci_ominus1 spin_square sum."""
    ss, multip = psi.fcisolver.spin_square(fcivec, psi.norb, psi.nelec)
    return float(ss), float(multip)


def run_uscc(mol, mf, las, a_idxs, i_idxs, verbose=0):
    """Run LAS-USCC; return (e_tot, psi)."""
    mc = mcscf.CASCI(mf, las.ncas, las.nelecas)
    mc.mo_coeff = las.mo_coeff
    mc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs, i_idxs)
    mc.fcisolver.norb_f = las.ncas_sub
    ci0_f = [np.squeeze(fockspace.hilbert2fock(ci[0], no, ne))
             for ci, no, ne in zip(las.ci, las.ncas_sub, las.nelecas_sub)]
    mc.fcisolver.get_init_guess = lambda *args: ci0_f
    mc.verbose = verbose
    mc.kernel()
    return mc.e_tot, mc.fcisolver.psi


def decompose_in_plane(LAS, LUSCC, USCC):
    def dot(a, b):
        return float(np.dot(a.ravel(), b.ravel()))

    n_LAS = np.sqrt(dot(LAS, LAS))
    n_LUSCC = np.sqrt(dot(LUSCC, LUSCC))
    n_USCC = np.sqrt(dot(USCC, USCC))

    LAS_n   = LAS   / n_LAS
    LUSCC_n = LUSCC / n_LUSCC
    USCC_n  = USCC  / n_USCC

    e1 = LAS_n
    proj_LUSCC_on_e1 = dot(e1, LUSCC_n)
    rem = LUSCC_n - proj_LUSCC_on_e1 * e1
    rem_norm = np.sqrt(dot(rem, rem))
    e2 = rem / rem_norm

    a = dot(e1, USCC_n)
    b = dot(e2, USCC_n)
    P_USCC_n = a * e1 + b * e2
    perp = USCC_n - P_USCC_n
    c = np.sqrt(max(dot(perp, perp), 0.0))

    theta_LUSCC = float(np.degrees(np.arctan2(rem_norm, proj_LUSCC_on_e1)))
    theta_USCC_in_plane = float(np.degrees(np.arctan2(b, a)))
    theta_USCC_out = float(np.degrees(np.arctan2(c, np.sqrt(a * a + b * b))))

    S = np.array([
        [1.0, dot(LAS_n, USCC_n), dot(LAS_n, LUSCC_n)],
        [dot(LAS_n, USCC_n), 1.0, dot(USCC_n, LUSCC_n)],
        [dot(LAS_n, LUSCC_n), dot(USCC_n, LUSCC_n), 1.0],
    ])

    return {
        "overlap": S,
        "norms": (n_LAS, n_USCC, n_LUSCC),
        "LAS_xyz": np.array([1.0, 0.0, 0.0]),
        "LUSCC_xyz": np.array([proj_LUSCC_on_e1, rem_norm, 0.0]),
        "USCC_xyz": np.array([a, b, c]),
        "PUSCC_xyz": np.array([a, b, 0.0]),
        "theta_LUSCC_deg": theta_LUSCC,
        "theta_USCC_in_plane_deg": theta_USCC_in_plane,
        "theta_USCC_out_deg": theta_USCC_out,
        "PUSCC_minus_LUSCC": float(np.linalg.norm(
            np.array([a, b, 0.0]) - np.array([proj_LUSCC_on_e1, rem_norm, 0.0]))),
        "out_of_plane_amp": float(c),
    }


def plot_one(coord, e_tot, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    LUS = coord["LUSCC_xyz"]
    USC = coord["USCC_xyz"]

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    SCH_B = 0.7
    in_plane_ratio = USC[1] / max(LUS[1], 1e-12)
    out_ratio = USC[2] / max(LUS[1], 1e-12)
    LAS_s   = np.array([1.0, 0.0, 0.0])
    LUSCC_s = np.array([np.sqrt(1 - SCH_B ** 2), SCH_B, 0.0])
    USCC_raw = np.array([1.0, SCH_B * in_plane_ratio, SCH_B * out_ratio])
    USCC_s   = USCC_raw / np.linalg.norm(USCC_raw)
    PUSCC_s  = np.array([USCC_s[0], USCC_s[1], 0.0])
    PUSCC_s /= np.linalg.norm(PUSCC_s)

    O = np.zeros(3)
    for color, v in [("k", LAS_s), ("tab:blue", LUSCC_s),
                     ("tab:red", USCC_s), ("tab:orange", PUSCC_s)]:
        ax.quiver(*O, *v, color=color, arrow_length_ratio=0.08, linewidth=2.2)
    ax.text(LAS_s[0] + 0.05, LAS_s[1] - 0.04, LAS_s[2] - 0.04,
            "|LAS>", color="k", fontsize=10)
    ax.text(LUSCC_s[0] - 0.05, LUSCC_s[1] + 0.06, LUSCC_s[2],
            "|LAS_LUSCC>", color="tab:blue", fontsize=10)
    ax.text(USCC_s[0] - 0.10, USCC_s[1] + 0.02, USCC_s[2] + 0.04,
            "|LAS_USCC>", color="tab:red", fontsize=10)
    ax.text(PUSCC_s[0] + 0.02, PUSCC_s[1] + 0.04, PUSCC_s[2] - 0.05,
            "P|LAS_USCC>", color="tab:orange", fontsize=10)
    plane = np.array([O, LAS_s * 1.05, LUSCC_s * 1.05])
    ax.plot_trisurf(plane[:, 0], plane[:, 1], plane[:, 2],
                    alpha=0.10, color="tab:blue")
    ax.plot(*zip(PUSCC_s, USCC_s), color="tab:red", ls=":", lw=1.3)
    ax.plot(*zip(PUSCC_s, LUSCC_s), color="tab:orange", ls=":", lw=1.3)
    ax.set_xlim(0, 1.1); ax.set_ylim(0, 1.0); ax.set_zlim(0, 0.5)
    ax.set_xlabel("e1  (||  |LAS>)")
    ax.set_ylabel("e2  (in-plane)")
    ax.set_zlabel("e3  (out-of-plane)")
    ax.set_title("Schematic (angles exaggerated)")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.axvline(0, color="gray", lw=0.5)
    ax2.plot(0, 0, "ko", ms=8)
    ax2.annotate("|LAS>", (0, 0), xytext=(6, 6), textcoords="offset points")
    ax2.plot(LUS[1], 0, "o", color="tab:blue", ms=8)
    ax2.annotate("|LAS_LUSCC>", (LUS[1], 0), xytext=(6, 6),
                 textcoords="offset points", color="tab:blue")
    ax2.plot(USC[1], USC[2], "o", color="tab:red", ms=8)
    ax2.annotate("|LAS_USCC>", (USC[1], USC[2]), xytext=(6, 6),
                 textcoords="offset points", color="tab:red")
    ax2.plot(USC[1], 0, "o", color="tab:orange", ms=8)
    ax2.annotate("P|LAS_USCC>", (USC[1], 0), xytext=(6, -14),
                 textcoords="offset points", color="tab:orange")
    ax2.plot([USC[1], USC[1]], [0, USC[2]], color="tab:red", ls=":", lw=1)
    ax2.plot([USC[1], LUS[1]], [0, 0], color="tab:orange", ls=":", lw=1)
    margin = max(abs(LUS[1]), abs(USC[1]), abs(USC[2])) * 1.4 + 1e-6
    ax2.set_xlim(-margin * 0.2, margin)
    ax2.set_ylim(-margin * 0.2, margin)
    ax2.set_aspect("equal")
    ax2.set_xlabel("e2  (in-plane, perp. to |LAS>)")
    ax2.set_ylabel("e3  (out-of-plane)")
    ax2.set_title("True deviation from |LAS>")
    ax2.grid(True, alpha=0.3)
    numbers = (
        f"theta(LAS,LUSCC)        = {coord['theta_LUSCC_deg']:.4f} deg\n"
        f"USCC out-of-plane angle = {coord['theta_USCC_out_deg']:.4f} deg\n"
        f"out-of-plane amplitude  = {coord['out_of_plane_amp']:.3e}\n"
        f"|| P|USCC> - |LUSCC> || = {coord['PUSCC_minus_LUSCC']:.3e}\n"
        f"E(USCC)  = {e_tot['USCC']:.6f}\n"
        f"E(LUSCC) = {e_tot['LUSCC']:.6f}\n"
        f"E(LAS)   = {e_tot['LAS']:.6f}"
    )
    ax2.text(0.02, 0.98, numbers, transform=ax2.transAxes,
             ha="left", va="top", family="monospace", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_sweep(records, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fracs = [r["frac"] for r in records]
    out_amp = [r["coord"]["out_of_plane_amp"] for r in records]
    delta = [r["coord"]["PUSCC_minus_LUSCC"] for r in records]
    theta = [r["coord"]["theta_LUSCC_deg"] for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, ys, ylab, t in [
        (axes[0], out_amp, "|out-of-plane| of |USCC>",       "USCC out-of-plane amplitude"),
        (axes[1], delta,   "|| P|USCC> - |LUSCC> ||",         "In-plane distance P|USCC> vs |LUSCC>"),
        (axes[2], theta,   "angle |LAS>-|LUSCC> (deg)",       "|LAS>-|LUSCC> opening angle"),
    ]:
        ax.plot(fracs, ys, "o-")
        ax.set_xscale("log")
        ax.set_xlabel("FRAC")
        ax.set_ylabel(ylab)
        ax.set_title(t)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_comparison(label, mol, mf, las, fracs, out_dir, title_prefix):
    """Run the USCC-vs-LUSCC comparison across FRACs; save figures + npz.

    Args:
        label:         filename stem ("h4", "chn_10_10", ...)
        mol, mf, las:  pre-built molecule, RHF, LASSCF objects
        fracs:         iterable of operator-selection fractions
        out_dir:       Path for outputs
        title_prefix:  text for figure suptitles, e.g. "H4 sto-3g"
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"RHF/mf    = {type(mf).__name__}  e = {mf.e_tot:.10f}")
    print(f"mol.spin  = {mol.spin}   (mol.nelec total = {mol.nelectron})")
    print(f"LASSCF    = {las.e_tot:.10f}")
    print(f"las.ncas       = {las.ncas}")
    print(f"las.ncas_sub   = {tuple(las.ncas_sub)}")
    print(f"las.nelecas    = {las.nelecas}  -> 2S = {las.nelecas[0]-las.nelecas[1]}")
    print(f"las.nelecas_sub= {[tuple(n) for n in las.nelecas_sub]}")

    _, g_sel_all, a_all, i_all = grad.get_grad_exact(las, epsilon=0.0)
    g_all = np.array(g_sel_all)[:, 0]
    print(f"total excitations available: {len(a_all)}")

    mc_for_h = mcscf.CASCI(mf, las.ncas, las.nelecas)
    mc_for_h.mo_coeff = las.mo_coeff
    h1, ecore = mc_for_h.get_h1eff()
    h2 = mc_for_h.get_h2eff()
    h_eff = [ecore, h1, h2]

    records = []
    for frac in fracs:
        print(f"\n=== FRAC = {frac} ===")
        a_sel, i_sel, g_sel = get_sorted_excitations(a_all, i_all, g_all, fraction=frac)
        print(f"  operators = {len(a_sel)}")
        t0 = time.time()

        fcisolver_dummy = lasuccsd.FCISolver_USCC(mol, a_sel, i_sel)
        fcisolver_dummy.norb_f = las.ncas_sub
        ci0_f_LAS = [np.squeeze(fockspace.hilbert2fock(ci[0], no, ne))
                     for ci, no, ne in zip(las.ci, las.ncas_sub, las.nelecas_sub)]
        psi_LAS = fcisolver_dummy.build_psi(
            ci0_f_LAS, las.ncas, las.ncas_sub, sum(las.nelecas), frozen=None)
        LAS_fci = psi_LAS.dp_ci(ci0_f_LAS)
        e_LAS_check = energy_of(psi_LAS, h_eff, LAS_fci)
        ss_LAS, mult_LAS = spin_square_of(psi_LAS, LAS_fci)
        print(f"  <LAS|H|LAS>   = {e_LAS_check:.10f}   (las.e_tot = {las.e_tot:.10f})")
        print(f"  <LAS|S^2|LAS> = {ss_LAS:.6f}   multip = {mult_LAS:.3f}")
        assert abs(e_LAS_check - las.e_tot) < 1e-6, "LAS energy mismatch"

        t_uscc = time.time()
        e_uscc, psi_uscc = run_uscc(mol, mf, las, a_sel, i_sel)
        USCC_fci = psi_uscc.get_fcivec(psi_uscc.x)
        e_USCC_check = energy_of(psi_uscc, h_eff, USCC_fci)
        ss_USCC, mult_USCC = spin_square_of(psi_uscc, USCC_fci)
        print(f"  USCC e_tot    = {e_uscc:.10f}   "
              f"(check {e_USCC_check:.10f}, took {time.time()-t_uscc:.1f}s)")
        print(f"  <USCC|S^2|USCC> = {ss_USCC:.6f}   multip = {mult_USCC:.3f}")
        assert abs(e_USCC_check - e_uscc) < 1e-5, "USCC energy reconstruction mismatch"

        t_luscc = time.time()
        lsi = LSI_LUSCC(las, a_sel, i_sel)
        lsi.verbose = 0
        e_roots_lassi, _ = lsi.kernel()
        e_luscc_lassi = float(e_roots_lassi[0])
        vecs = build_rootspace_fci_vectors(lsi, psi_LAS)
        eigvals, eigvecs = solve_luscc_in_fock_basis(vecs, psi_LAS, h_eff)
        e_luscc = float(eigvals[0])
        c0 = eigvecs[:, 0]
        LUSCC_fci = np.zeros_like(LAS_fci)
        for r, v in enumerate(vecs):
            LUSCC_fci += c0[r] * v
        if np.dot(LAS_fci.ravel(), LUSCC_fci.ravel()) < 0:
            LUSCC_fci = -LUSCC_fci
        if np.dot(LAS_fci.ravel(), USCC_fci.ravel()) < 0:
            USCC_fci = -USCC_fci
        e_LUSCC_check = energy_of(psi_LAS, h_eff, LUSCC_fci)
        ss_LUSCC, mult_LUSCC = spin_square_of(psi_LAS, LUSCC_fci)
        n_LUSCC = float(np.linalg.norm(LUSCC_fci.ravel()))
        print(f"  LUSCC e_root  = {e_luscc:.10f} (LASSI {e_luscc_lassi:.10f}, "
              f"||LUSCC||={n_LUSCC:.6f}, took {time.time()-t_luscc:.1f}s)")
        print(f"  <LUSCC|S^2|LUSCC> = {ss_LUSCC:.6f}   multip = {mult_LUSCC:.3f}")
        assert abs(e_luscc - e_luscc_lassi) < 1e-5, \
            "LUSCC eigenvalues differ between LASSI and Fock-rebuilt basis"
        assert abs(e_LUSCC_check - e_luscc) < 1e-6, "LUSCC energy reconstruction mismatch"

        coord = decompose_in_plane(LAS_fci, LUSCC_fci, USCC_fci)
        print("  overlap (LAS, USCC, LUSCC):")
        for row in coord["overlap"]:
            print("    " + "  ".join(f"{v:+.8f}" for v in row))
        print(f"  theta(LAS,LUSCC)        = {coord['theta_LUSCC_deg']:.4f} deg")
        print(f"  USCC out-of-plane angle = {coord['theta_USCC_out_deg']:.4f} deg")
        print(f"  USCC out-of-plane amp   = {coord['out_of_plane_amp']:.6e}")
        print(f"  || P|USCC> - |LUSCC> || = {coord['PUSCC_minus_LUSCC']:.6e}")
        print(f"  total time              = {time.time()-t0:.1f}s")

        np.savez(
            out_dir / f"{label}_frac{frac:.2f}.npz",
            frac=frac, n_operators=len(a_sel),
            e_las=las.e_tot, e_uscc=e_uscc, e_luscc=e_luscc,
            overlap=coord["overlap"],
            LAS_xyz=coord["LAS_xyz"], LUSCC_xyz=coord["LUSCC_xyz"],
            USCC_xyz=coord["USCC_xyz"], PUSCC_xyz=coord["PUSCC_xyz"],
            uscc_x=np.asarray(psi_uscc.x),
            a_sel=np.array(a_sel, dtype=object),
            i_sel=np.array(i_sel, dtype=object),
            gradients=np.asarray(g_sel),
        )
        plot_one(coord,
                 e_tot={"LAS": las.e_tot, "USCC": e_uscc, "LUSCC": e_luscc},
                 title=f"{title_prefix}, FRAC={frac}, n_op={len(a_sel)}",
                 out_path=out_dir / f"{label}_frac{frac:.2f}.png")
        records.append({"frac": frac, "coord": coord,
                        "e_las": las.e_tot, "e_uscc": e_uscc, "e_luscc": e_luscc})

    plot_sweep(records, out_dir / f"{label}_sweep.png")
    print(f"\nWrote per-frac data and figures to {out_dir}/")
