"""Stilbene at 90 deg dihedral, lowest TRIPLET state. (10,10) active space, 6-31g.

Triplet on central C=C (frag 1, diradical at 90 deg); singlet on phenyls.
ROHF + mol.spin=2.
"""

from pathlib import Path
from pyscf import gto, scf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

from tasks.uscc_luscc.compare import run_comparison


PWD = Path(__file__).resolve().parent
DATA = PWD / "data"
GEOM = PWD.parent / "geom" / "stil90.xyz"


def build_las():
    xyz = GEOM.read_text()
    mol = gto.M(atom=xyz, basis="6-31g", spin=2, verbose=0, max_memory=70_000)
    mf = scf.ROHF(mol).run()
    las = LASSCF(mf, (4, 2, 4), (4, 2, 4), spin_sub=(1, 3, 1), verbose=0)
    las.max_memory = 70_000
    frag_atom_list = [
        [1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19],
        [0, 7, 14, 20],
        [8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25],
    ]
    mo_loc = las.localize_init_guess(frag_atom_list, mf.mo_coeff)
    las.kernel(mo_loc)
    return mol, mf, las


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    mol, mf, las = build_las()
    run_comparison(
        label="stil90_triplet",
        mol=mol, mf=mf, las=las,
        fracs=(0.01, 0.02, 0.03, 0.04),
        out_dir=DATA,
        title_prefix="stilbene 90 deg triplet (10,10)",
    )


if __name__ == "__main__":
    main()
