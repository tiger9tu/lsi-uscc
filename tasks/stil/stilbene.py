import numpy as np
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.citools import grad
from mrh.my_pyscf import lassi
from lcc import LSI_LUSCC
from helper.util import get_sorted_excitations
from pathlib import Path
from time import time

# ── 1. System parameters ───────────────────────────────────────────────────
# Set DIHEDRAL to the desired C=C dihedral angle: 1, 60, 90, 120, or 180
DIHEDRAL = 90
FRACS = [0.01, 0.02, 0.03, 0.04]

ncas_f      = (4, 2, 4)
nelecas_f   = (4, 2, 4)
spin_sub_f  = (1, 1, 1)
frag_atom_list = [
    [1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19],
    [0, 7, 14, 20],
    [8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25],
]
basis   = '6-31g'
verbose = 3

pwd = Path(__file__).resolve().parent
geom_name = f'stil{DIHEDRAL:03d}' if DIHEDRAL == 1 else f'stil{DIHEDRAL}'
geom_path = pwd / 'geom' / f'{geom_name}.xyz'
log_path  = pwd / 'data'  / f'{geom_name}.log'
with geom_path.open('r') as f:
    xyz = f.read()

# ── 2. Molecule + RHF ──────────────────────────────────────────────────────
mol = gto.M(atom=xyz, basis=basis, output=str(log_path), verbose=verbose)
mf = scf.RHF(mol).run()

# Reference CASCI
cas = mcscf.CASCI(mf, sum(ncas_f), sum(nelecas_f)).run()
print("CASCI energy =", cas.e_tot)

# ── 3. LASSCF ──────────────────────────────────────────────────────────────
las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f, verbose=verbose)
mo_loc = las.localize_init_guess(frag_atom_list, mf.mo_coeff)
las.kernel(mo_loc)
print("LASSCF energy =", las.e_tot)

# ── 4. LASSIS (reference) ──────────────────────────────────────────────────
lsi = lassi.LASSIS(las)
t0 = time()
e_lsi, _ = lsi.kernel()
print("LASSIS energy =", e_lsi[0])
print("LASSIS time: {:.2f} s".format(time() - t0))

# ── 5. Excitation selection ────────────────────────────────────────────────
_, g_sel, a_idxs_all, i_idxs_all = grad.get_grad_exact(las, epsilon=0.0)
g_all = np.array(g_sel)[:, 0]

# ── 6. LSI_LUSCC (sweep over fractions) ────────────────────────────────────
for frac in FRACS:
    a_idxs, i_idxs, _ = get_sorted_excitations(
        a_idxs_all, i_idxs_all, g_all, fraction=frac)
    print(f"\nFraction: {frac} | Number of excitations: {len(a_idxs)}")
    t0 = time()
    lsi_luscc = LSI_LUSCC(las, a_idxs, i_idxs)
    e_roots, si = lsi_luscc.kernel()
    print("LSI-LUSCC ground state energy =", e_roots[0])
    print("Time: {:.2f} s".format(time() - t0))
