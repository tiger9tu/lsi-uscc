from pathlib import Path

import numpy as np
from pyscf import gto, lib, scf
from pyscf.tools import molden
from mrh.my_pyscf.mcscf import lasscf_async as asyn


PWD = Path(__file__).resolve().parent
DATA = PWD / "data"
SOURCE = PWD / "source" / "paper_data" / "chan" / "2220_4frag"

BASIS = {
    "C": "cc-pvdz",
    "H": "cc-pvdz",
    "S": "aug-cc-pvdz",
    "Fe": "aug-cc-pvdz",
}
NCAS_F = (5, 5, 5, 5)
NELECAS_F = ((5, 1), (5, 0), (1, 5), (0, 5))
SPIN_SUB = (5, 6, 5, 6)
FRAG_ATOM_LIST = ([0], [1], [2], [3])
MO_LIST = [
    125, 126, 127, 128, 129,
    135, 136, 137, 138, 139,
    145, 146, 147, 148, 149,
    155, 156, 157, 158, 159,
]


def ensure_data():
    DATA.mkdir(parents=True, exist_ok=True)


def build_mf(output_name, max_memory):
    ensure_data()
    lib.logger.TIMER_LEVEL = lib.logger.INFO
    mol = gto.M(
        atom=str(SOURCE / "geom.xyz"),
        verbose=4,
        spin=18,
        charge=-2,
        basis=BASIS,
        max_memory=max_memory,
        output=str(DATA / output_name),
    )
    mf = scf.ROHF(mol)
    mf.init_guess = "atom"
    mf = mf.density_fit()
    mf = mf.newton()
    mf.kernel()
    print(f"ROHF energy = {mf.e_tot}", flush=True)
    np.save(DATA / "fe4_rohf_mo_coeff.npy", mf.mo_coeff)
    return mol, mf


def build_deposited_las(mf, max_memory):
    las = asyn.LASSCF(
        mf,
        NCAS_F,
        NELECAS_F,
        spin_sub=SPIN_SUB,
        verbose=4,
        assert_no_dupes=False,
    )
    las.max_memory = max_memory
    las.chkfile = str(DATA / "fe4_las.chk")
    las.set_fragments_(FRAG_ATOM_LIST, mf.mo_coeff, mo_occ=mf.mo_occ)
    _, _, conv_mo_coeff, _, _, _ = molden.load(str(SOURCE / "converged.molden"))
    mo_coeff = las.sort_mo(MO_LIST, conv_mo_coeff)
    return las, mo_coeff
