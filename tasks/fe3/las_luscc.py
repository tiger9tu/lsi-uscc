"""Run LAS and LAS-LUSCC for the Fe3 input in this directory.

The molecule, basis, HF, AVAS, and LASSCF setup follow Joanna-input.py. This
driver intentionally does not run USCC.
"""

import argparse
import sys
from pathlib import Path
from time import time

import numpy as np
from pyscf import gto, scf, lib, mcscf
from pyscf.mcscf import avas
from mrh.exploratory.citools import grad
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.tools import molden

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
ACTIVE_LIST = list(range(8, 23))
FRAG_ATOM_LIST = ([17], [19], [22])


def read_xyz_atom(path):
    atoms = []
    for line in path.read_text().splitlines():
        fields = line.replace(",", " ").split()
        if len(fields) == 4:
            atoms.append(" ".join(fields))
    return "\n".join(atoms)


def build_las(max_memory):
    DATA.mkdir(parents=True, exist_ok=True)
    lib.logger.TIMER_LEVEL = lib.logger.INFO

    mol = gto.M(
        atom=read_xyz_atom(GEOM),
        verbose=4,
        spin=10,
        charge=0,
        basis=BASIS,
        output=str(DATA / "lasscf_hf.log"),
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

    np.save(DATA / "hf_is.npy", mf.mo_coeff)
    print(f"ROHF energy = {mf.e_tot}", flush=True)

    ncas, nelecas, guess_mo_coeff = avas.kernel(
        mf, ["Fe 3d", "Fe 4d"], minao=mol.basis, openshell_option=3)
    print(f"AVAS ncas={ncas} nelecas={nelecas}", flush=True)
    molden.from_mo(mol, str(DATA / "avas.molden"), guess_mo_coeff)

    mc_test = mcscf.CASCI(mf, ncas, nelecas)
    final_list = [idx + mc_test.ncore for idx in ACTIVE_LIST]
    print(f"final active orbital list = {final_list}", flush=True)

    las = LASSCF(
        mf, (5, 5, 5), ((5, 1), (4, 1), (4, 1)),
        spin_sub=(5, 4, 4), verbose=4)
    las.max_memory = max_memory
    las.max_cycle_macro = 200

    mo_sorted = las.sort_mo(final_list, guess_mo_coeff)
    mo_localized = las.localize_init_guess(FRAG_ATOM_LIST, mo_sorted)

    las_guess = DATA / "las_15_16_is.npy"
    if las_guess.exists():
        print(f"using saved LAS MO initial guess {las_guess}", flush=True)
        mo_localized = np.load(las_guess)

    t0 = time()
    las.kernel(mo_localized)
    print(f"LASSCF energy = {las.e_tot}", flush=True)
    print(f"LASSCF time: {time() - t0:.2f} s", flush=True)

    np.save(DATA / "las_15_16_is.npy", las.mo_coeff)
    molden.from_lasscf(las, str(DATA / "las_15_16_is.molden"))
    return mol, mf, las


def run_luscc(las, frac, epsilon):
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
    las_luscc = LSI_LUSCC(las, a_idxs, i_idxs)
    e_roots, si = las_luscc.kernel()
    si_arr = np.asarray(si)
    si0 = si_arr[:, 0]
    print(f"LAS-LUSCC time: {time() - t0:.2f} s", flush=True)
    print(f"LAS-LUSCC ground state energy = {e_roots[0]}", flush=True)
    print(f"LAS-LUSCC number of LUSCC basis states: {si_arr.shape[0]}", flush=True)
    print(f"LAS-LUSCC number of roots: {len(e_roots)}", flush=True)

    np.save(DATA / f"fe3_{selection_label}_e.npy", np.asarray(e_roots))
    np.save(DATA / f"fe3_{selection_label}_si0.npy", si0)
    np.save(DATA / f"fe3_{selection_label}_grads_sel.npy", np.asarray(selected_grads))
    np.save(DATA / f"fe3_{selection_label}_a_idxs.npy",
            np.array(a_idxs, dtype=object), allow_pickle=True)
    np.save(DATA / f"fe3_{selection_label}_i_idxs.npy",
            np.array(i_idxs, dtype=object), allow_pickle=True)

    top = np.argsort(-np.abs(si0))[:10]
    print(f"Top-10 |coeff| basis indices: {top.tolist()}", flush=True)
    print(f"Top-10 coefficients: {si0[top].tolist()}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frac", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--max-memory", type=int, default=256000)
    args = parser.parse_args()

    _, _, las = build_las(args.max_memory)
    run_luscc(las, args.frac, args.epsilon)


if __name__ == "__main__":
    main()
