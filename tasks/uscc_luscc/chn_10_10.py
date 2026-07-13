"""Compare |LAS_USCC> and |LAS_LUSCC> in the FCI basis on C2H4N4 (bisdiazene)
with a (10,10) active space, 6-31g basis. Fragments (4,2,4) orbitals.

Larger than H4: ncas=10 -> Fock-space FCI shape (1024, 1024) (~1 M float64
per state). Operator counts scale up quickly; we sweep small FRACs only.
"""

from pathlib import Path
from pyscf import scf, lib
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.tests.lasscf.c2h4n4_struct import structure as struct

from tasks.uscc_luscc.compare import run_comparison


PWD = Path(__file__).resolve().parent
DATA = PWD / "data"


def build_las():
    lib.logger.TIMER_LEVEL = lib.logger.INFO
    mol = struct(2.0, 2.0, "6-31g")
    mol.output = str(DATA / "chn_10_10_631g.log")
    mol.verbose = 3
    mol.spin = 8
    mol.max_memory = 60_000  # MB
    mol.build()
    mf = scf.RHF(mol).run()

    ncas_f = (4, 2, 4)
    nelecas_f = ((2, 2), (1, 1), (2, 2))
    spin_sub = (1, 1, 1)
    frag_atom_list = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]

    las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub)
    mo_coeff = las.localize_init_guess(frag_atom_list)
    las.kernel(mo_coeff)
    return mol, mf, las


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    mol, mf, las = build_las()
    run_comparison(
        label="chn_10_10",
        mol=mol, mf=mf, las=las,
        fracs=(0.01, 0.02, 0.03, 0.04),
        out_dir=DATA,
        title_prefix="C2H4N4 (10,10) 6-31g",
    )


if __name__ == "__main__":
    main()
