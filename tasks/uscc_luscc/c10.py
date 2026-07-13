"""Compare |LAS_USCC> and |LAS_LUSCC> on polyene C10 (6-31g, 5 fragments).

Active space: (10,10) with 5 fragments of (2,2) electrons in 2 orbitals each.
ncas=10 -> Fock-space FCI shape (1024, 1024) (~8 MB / state). Operator counts
scale up, so we only run the single FRAC the user requested.
"""

from pathlib import Path
from pyscf import gto, scf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

from tasks.uscc_luscc.compare import run_comparison


PWD = Path(__file__).resolve().parent
DATA = PWD / "data"
GEOM = PWD.parent / "geom" / "c10.xyz"


def build_las():
    xyz = GEOM.read_text()
    mol = gto.M(atom=xyz, basis="6-31g", verbose=0)
    mf = scf.RHF(mol).run()
    las = LASSCF(mf,
                 (2, 2, 2, 2, 2), (2, 2, 2, 2, 2),
                 spin_sub=(1, 1, 1, 1, 1), verbose=0)
    mo_loc = las.localize_init_guess(
        [[0, 2], [10, 12], [18, 19], [13, 11], [3, 1]], mf.mo_coeff)
    las.kernel(mo_loc)
    return mol, mf, las


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    mol, mf, las = build_las()
    run_comparison(
        label="c10",
        mol=mol, mf=mf, las=las,
        fracs=(0.02,),
        out_dir=DATA,
        title_prefix="polyene C10 6-31g",
    )


if __name__ == "__main__":
    main()
