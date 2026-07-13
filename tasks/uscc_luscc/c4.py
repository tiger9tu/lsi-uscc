"""Compare |LAS_USCC> and |LAS_LUSCC> on polyene C4 (6-31g, 2 fragments)."""

from pathlib import Path
from pyscf import gto, scf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

from tasks.uscc_luscc.compare import run_comparison


PWD = Path(__file__).resolve().parent
DATA = PWD / "data"
GEOM = PWD.parent / "geom" / "c4.xyz"


def build_las():
    xyz = GEOM.read_text()
    mol = gto.M(atom=xyz, basis="6-31g", verbose=0)
    mf = scf.RHF(mol).run()
    las = LASSCF(mf, (2, 2), (2, 2), spin_sub=(1, 1), verbose=0)
    mo_loc = las.localize_init_guess(((0, 2), (3, 1)), mf.mo_coeff)
    las.kernel(mo_loc)
    return mol, mf, las


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    mol, mf, las = build_las()
    run_comparison(
        label="c4",
        mol=mol, mf=mf, las=las,
        fracs=(0.01, 0.02, 0.03, 0.04),
        out_dir=DATA,
        title_prefix="polyene C4 6-31g",
    )


if __name__ == "__main__":
    main()
