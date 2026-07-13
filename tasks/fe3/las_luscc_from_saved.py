"""Run Fe3 LAS-LUSCC from saved LASSCF orbitals without orbital reoptimization."""

import argparse
import sys
from pathlib import Path
from time import time

import numpy as np
from pyscf import gto, scf, lib
from mrh.exploratory.citools import grad
from mrh.my_pyscf.mcscf.chkfile import dump_las, load_las_
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

from helper.util import get_sorted_excitations
from lcc import LSI_LUSCC


PWD = Path(__file__).resolve().parent
DATA = PWD / "data"
GEOM = PWD / "fe3.xyz"

BASIS = {
    "C": "cc-pvdz",
    "H": "cc-pvdz",
    "O": "cc-pvtz",
    "Al": "cc-pvtz",
    "Fe": "cc-pvtz",
}
NCAS_F = (5, 5, 5)
NELECAS_F = ((5, 1), (4, 1), (4, 1))
SPIN_SUB = (5, 4, 4)


def read_xyz_atom(path):
    atoms = []
    for line in path.read_text().splitlines():
        fields = line.replace(",", " ").split()
        if len(fields) == 4:
            atoms.append(" ".join(fields))
    return "\n".join(atoms)


def build_mf(max_memory):
    DATA.mkdir(parents=True, exist_ok=True)
    lib.logger.TIMER_LEVEL = lib.logger.INFO

    mol = gto.M(
        atom=read_xyz_atom(GEOM),
        verbose=4,
        spin=10,
        charge=0,
        basis=BASIS,
        output=str(DATA / "fe3_luscc_from_saved.log"),
        max_memory=max_memory,
    )

    mf = scf.ROHF(mol)
    mf.init_guess = "atom"
    mf = mf.density_fit()
    mf.max_cycle = 1
    mf.kernel()
    mf = mf.newton()
    mf.max_cycle = 1

    hf_is = DATA / "hf_is.npy"
    if hf_is.exists():
        print(f"loading {hf_is}", flush=True)
        mf.mo_coeff = np.load(hf_is)
    mf.kernel()
    print(f"ROHF energy = {mf.e_tot}", flush=True)
    return mol, mf


def build_las_from_saved(mf, max_memory):
    chk_path = DATA / "fe3_las.chk"
    las = LASSCF(mf, NCAS_F, NELECAS_F, spin_sub=SPIN_SUB, verbose=4)
    las.max_memory = max_memory
    las.max_cycle_macro = 200
    las.chkfile = str(chk_path)

    if chk_path.exists():
        print(f"loading LAS checkpoint {chk_path}", flush=True)
        load_las_(las, str(chk_path))
        print(f"LASCI/LASSCF energy = {las.e_tot}", flush=True)
        return las

    las_mo_path = DATA / "las_15_16_is.npy"
    if not las_mo_path.exists():
        raise FileNotFoundError(f"Missing saved LASSCF orbitals: {las_mo_path}")
    las_mo = np.load(las_mo_path)
    print(f"loaded LASSCF orbitals from {las_mo_path}", flush=True)

    t0 = time()
    las.lasci(mo_coeff=las_mo)
    print(f"fixed-orbital LASCI energy = {las.e_tot}", flush=True)
    print(f"fixed-orbital LASCI time: {time() - t0:.2f} s", flush=True)
    dump_las(las, str(chk_path))
    print(f"stored LAS checkpoint {chk_path}", flush=True)
    return las


def run_luscc(las, frac, epsilon, lindep_thresh, norm_thresh):
    t0 = time()
    _, g_sel, a_idxs_all, i_idxs_all = grad.get_grad_exact(las, epsilon=0.0)
    g_all = np.array(g_sel)[:, 0]
    print(f"get_grad_exact time: {time() - t0:.2f} s", flush=True)
    print(f"total excitations available: {len(a_idxs_all)}", flush=True)

    if epsilon is None:
        a_idxs, i_idxs, selected_grads = get_sorted_excitations(
            a_idxs_all, i_idxs_all, g_all, fraction=frac)
        selection_label = f"frac{frac:.4f}"
        print(f"selection fraction = {frac}", flush=True)
    else:
        a_idxs, i_idxs, selected_grads = get_sorted_excitations(
            a_idxs_all, i_idxs_all, g_all, epsilon=epsilon)
        selection_label = f"eps{epsilon:.6g}".replace(".", "p")
        print(f"selection epsilon = {epsilon}", flush=True)

    print(f"selected excitations: {len(a_idxs)}", flush=True)
    sys.stdout.flush()

    t0 = time()
    las_luscc = LSI_LUSCC(
        las, a_idxs, i_idxs,
        lindep_thresh=lindep_thresh,
        norm_thresh=norm_thresh)
    e_roots, si = las_luscc.kernel()
    si_arr = np.asarray(si)
    si0 = si_arr[:, 0]
    print(f"LAS-LUSCC time: {time() - t0:.2f} s", flush=True)
    print(f"LAS-LUSCC ground state energy = {e_roots[0]}", flush=True)
    print(f"LAS-LUSCC number of LUSCC basis states: {si_arr.shape[0]}", flush=True)
    print(f"LAS-LUSCC number of roots: {len(e_roots)}", flush=True)

    np.save(DATA / f"fe3_savedlas_{selection_label}_e.npy", np.asarray(e_roots))
    np.save(DATA / f"fe3_savedlas_{selection_label}_si0.npy", si0)
    np.save(DATA / f"fe3_savedlas_{selection_label}_grads_sel.npy",
            np.asarray(selected_grads))
    np.save(DATA / f"fe3_savedlas_{selection_label}_a_idxs.npy",
            np.array(a_idxs, dtype=object), allow_pickle=True)
    np.save(DATA / f"fe3_savedlas_{selection_label}_i_idxs.npy",
            np.array(i_idxs, dtype=object), allow_pickle=True)

    top = np.argsort(-np.abs(si0))[:10]
    print(f"Top-10 |coeff| basis indices: {top.tolist()}", flush=True)
    print(f"Top-10 coefficients: {si0[top].tolist()}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frac", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--lindep-thresh", type=float, default=1e-4)
    parser.add_argument("--norm-thresh", type=float, default=1e-12)
    parser.add_argument("--max-memory", type=int, default=120000)
    args = parser.parse_args()

    _, mf = build_mf(args.max_memory)
    las = build_las_from_saved(mf, args.max_memory)
    run_luscc(las, args.frac, args.epsilon, args.lindep_thresh, args.norm_thresh)


if __name__ == "__main__":
    main()
