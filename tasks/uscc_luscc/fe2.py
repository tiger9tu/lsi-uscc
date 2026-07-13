"""Compare |LAS_USCC> and |LAS_LUSCC> on the Fe2 instance from lassqd_code.

The molecular and LASSCF setup follows FeFe_example/input.py in
https://github.com/joannaqw/lassqd_code:

  - Fe2 complex from fefe.xyz
  - ROHF, charge=4, spin=0, 6-31g
  - density fitting and atom initial guess
  - extremeAsynLASSCF with two (10,10) fragments and ((4,2),(2,4)) electrons

The upstream script loads fefe_as.npy and as_increase_avas.npy. This task uses
those files if present in tasks/uscc_luscc/data, otherwise it computes the AVAS
guess from Fe 3d orbitals.
"""

from pathlib import Path

import numpy as np
from pyscf import gto, scf, lib
from pyscf.mcscf import avas
from mrh.my_pyscf.mcscf.lasscf_rdm2 import extremeAsynLASSCF

from tasks.uscc_luscc.compare import run_comparison


PWD = Path(__file__).resolve().parent
DATA = PWD / "data"
GEOM = PWD.parent / "geom" / "fefe.xyz"

BASIS_PAPER = {
    "Fe": "6-31g",
    "C": "6-31g",
    "H": "6-31g",
    "O": "6-31g",
    "N": "6-31g",
}
MO_LIST = list(range(100, 120))


def _load_array(name):
    path = DATA / name
    if path.exists():
        print(f"loading {path}")
        return np.load(path)
    return None


def build_las():
    lib.logger.TIMER_LEVEL = lib.logger.INFO
    mol = gto.M(
        atom=str(GEOM),
        verbose=4,
        spin=0,
        charge=4,
        basis=BASIS_PAPER,
        output=str(DATA / "fe2_631g.log"),
        max_memory=120_000,
    )
    mol.build()

    mf = scf.ROHF(mol)
    mf.init_guess = "atom"
    mf = mf.density_fit()

    fefe_as = _load_array("fefe_as.npy")
    if fefe_as is not None:
        mf.mo_coeff = fefe_as
    mf.kernel()

    guess_mo_coeff = _load_array("as_increase_avas.npy")
    if guess_mo_coeff is None:
        _, _, guess_mo_coeff = avas.kernel(mf, ["Fe 3d"], minao=mol.basis)

    las = extremeAsynLASSCF(mf, (10, 10), ((4, 2), (2, 4)), spin_sub=(3, 3))
    guess_mo_sorted = las.sort_mo(MO_LIST, guess_mo_coeff)
    mo_localized = las.localize_init_guess(([0], [1]), guess_mo_sorted)
    las.max_cycle_rdmjk = 0
    las.kernel(mo_localized)
    return mol, mf, las


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    mol, mf, las = build_las()
    run_comparison(
        label="fe2",
        mol=mol,
        mf=mf,
        las=las,
        fracs=(0.01,),
        out_dir=DATA,
        title_prefix="Fe2 6-31g",
    )


if __name__ == "__main__":
    main()
