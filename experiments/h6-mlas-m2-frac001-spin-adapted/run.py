"""Spin-adapted, uncontracted mLAS-LUSCC validation on H6/STO-3G."""

import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from time import time

import numpy as np
from pyscf import gto, scf
from mrh.exploratory.citools.grad import get_grad_exact_lassi
from mrh.my_pyscf import lassi
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

from helper.util import get_sorted_excitations
from lcc import LSI_LUSCC


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
M = 2
FRACTION = 0.01
MULTIPLICITIES = (1, 3, 5, 7)


def spin_data(si):
    s2 = float(np.asarray(si.s2).reshape(-1)[0])
    spin = float((np.sqrt(max(0.0, 1.0 + 4.0*s2)) - 1.0) / 2.0)
    return s2, spin


def atomic_json(path, payload):
    with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=path.name + ".",
            suffix=".tmp", delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main():
    case = int(os.environ.get(
        "CASE_INDEX", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    smult = MULTIPLICITIES[case]
    label = {
        1: "singlet", 3: "triplet", 5: "quintet", 7: "septet"
    }[smult]
    run_id = f"h6-{label}-m2-frac0p01-spinadapted-uncontracted"
    results = HERE / "results"
    logs = HERE / "logs"
    results.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    started = time()

    mol = gto.M(
        atom=(ROOT / "tasks" / "geom" / "h6.xyz").read_text(),
        basis="sto-3g", spin=0, verbose=3,
        output=str(logs / f"{run_id}.pyscf.log"))
    mf = scf.RHF(mol).run()
    las = LASSCF(
        mf, (2, 2, 2), (2, 2, 2), spin_sub=(1, 1, 1), verbose=3)
    mo = las.localize_init_guess(
        ((0, 1), (2, 3), (4, 5)), mf.mo_coeff)
    las.kernel(mo)

    full = lassi.LASSIS(las)
    full.sisolver.smult = smult
    full.sisolver.nroots = 1
    print(f"{run_id}: target-spin LASSIS", flush=True)
    t0 = time()
    e_full, si_full = full.kernel(davidson_only=True)
    t_full = time() - t0
    full_s2, full_spin = spin_data(si_full)

    mlas = LSI_LUSCC(
        full, [], [], state=0, top_m=M, smult_si=smult,
        internally_contracted=False, opt=1)
    mlas.sisolver.nroots = 1
    print(f"{run_id}: spin-adapted mLAS", flush=True)
    t0 = time()
    e_mlas, si_mlas = mlas.kernel()
    t_mlas = time() - t0
    mlas_s2, mlas_spin = spin_data(si_mlas)

    print(f"{run_id}: exact LSI gradient and 1% selection", flush=True)
    t0 = time()
    gradients, _, a_all, i_all = get_grad_exact_lassi(
        mlas, state=0, epsilon=0.0)
    a_sel, i_sel, g_sel = get_sorted_excitations(
        a_all, i_all, np.asarray(gradients), fraction=FRACTION)
    t_grad = time() - t0

    solver = LSI_LUSCC(
        mlas, a_sel, i_sel, state=0, top_m=M, smult_si=smult,
        internally_contracted=False, opt=1)
    solver.sisolver.nroots = 1
    print(f"{run_id}: spin-adapted uncontracted mLAS-LUSCC", flush=True)
    t0 = time()
    e_luscc, si_luscc = solver.kernel()
    t_luscc = time() - t0
    luscc_s2, luscc_spin = spin_data(si_luscc)

    target_s = (smult - 1) / 2
    target_s2 = target_s * (target_s + 1)
    payload = {
        "run_id": run_id,
        "status": "succeeded",
        "parameters": {
            "multiplicity": smult, "m": M,
            "effective_m": min(M, np.asarray(full.si).shape[0]),
            "excitation_fraction": FRACTION,
            "available_excitations": len(a_all),
            "selected_excitations": len(a_sel),
            "spin_adapted": True, "internally_contracted": False,
        },
        "las": {"energy_eh": float(las.e_tot)},
        "lassis": {"energy_eh": float(e_full[0]), "s2": full_s2,
                   "spin": full_spin, "elapsed_s": t_full},
        "mlas": {"energy_eh": float(e_mlas[0]), "s2": mlas_s2,
                 "spin": mlas_spin, "elapsed_s": t_mlas},
        "mlas_luscc": {"energy_eh": float(e_luscc[0]), "s2": luscc_s2,
                       "spin": luscc_spin, "elapsed_s": t_luscc},
        "gradient_elapsed_s": t_grad,
        "validation": {
            "finite": bool(np.isfinite([
                e_full[0], e_mlas[0], e_luscc[0],
                full_s2, mlas_s2, luscc_s2]).all()),
            "mlas_spin_pure_1e4": bool(abs(mlas_s2-target_s2) <= 1e-4),
            "luscc_spin_pure_1e4":
                bool(abs(luscc_s2-target_s2) <= 1e-4),
            "luscc_not_above_mlas":
                bool(float(e_luscc[0]) <= float(e_mlas[0]) + 1e-9),
        },
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                text=True).strip(),
            "git_dirty": bool(subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain"],
                text=True).strip()),
            "mrh_commit": subprocess.check_output(
                ["git", "-C", str(ROOT / "external" / "mrh"),
                 "rev-parse", "HEAD"], text=True).strip(),
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
            "started_unix": started, "ended_unix": time(),
        },
    }
    if not all(payload["validation"].values()):
        payload["status"] = "failed_validation"
    atomic_json(results / f"{run_id}.json", payload)
    with (results / f"{run_id}.npz.tmp").open("wb") as handle:
        np.savez(
            handle, gradients=np.asarray(gradients),
            selected_gradients=np.asarray(g_sel),
            a_idxs=np.asarray(a_sel, dtype=object),
            i_idxs=np.asarray(i_sel, dtype=object),
            e_full=np.asarray(e_full), s2_full=np.asarray(si_full.s2),
            e_mlas=np.asarray(e_mlas), s2_mlas=np.asarray(si_mlas.s2),
            e_luscc=np.asarray(e_luscc), s2_luscc=np.asarray(si_luscc.s2))
    os.replace(results / f"{run_id}.npz.tmp",
               results / f"{run_id}.npz")
    print("H6_MLAS_LUSCC_RESULT " + json.dumps(payload, sort_keys=True),
          flush=True)
    if payload["status"] != "succeeded":
        raise RuntimeError(f"validation failed: {payload['validation']}")


if __name__ == "__main__":
    main()
