"""Compare |LAS_USCC> and |LAS_LUSCC> in the FCI basis on H4 sto-3g."""

from pathlib import Path
from pyscf import gto, scf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

from tasks.uscc_luscc.compare import run_comparison


PWD = Path(__file__).resolve().parent
DATA = PWD / "data"
GEOM = PWD.parent / "geom" / "h4.xyz"


def build_las():
    xyz = GEOM.read_text()
    mol = gto.M(atom=xyz, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    las = LASSCF(mf, (2, 2), (2, 2), spin_sub=(1, 1), verbose=0)
    mo_loc = las.localize_init_guess(((0, 1), (2, 3)), mf.mo_coeff)
    las.kernel(mo_loc)
    return mol, mf, las


def main():
    mol, mf, las = build_las()
    run_comparison(
        label="h4",
        mol=mol, mf=mf, las=las,
        fracs=(0.05, 0.1, 0.2, 0.5, 1.0),
        out_dir=DATA,
        title_prefix="H4 sto-3g",
    )


if __name__ == "__main__":
    main()
