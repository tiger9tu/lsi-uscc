import numpy as np
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.citools import grad
from lcc import LSI_LUSCC
from helper.util import get_sorted_excitations
from pathlib import Path

# ── 1. System parameters ───────────────────────────────────────────────────
FRACS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

ncas_f      = (2, 2, 2, 2)
nelecas_f   = (2, 2, 2, 2)
spin_sub_f  = (1, 1, 1, 1)
frag_atom_list = ((0, 1), (2, 3), (4, 5), (6, 7))

pwd = Path(__file__).resolve().parent
geom_path = pwd.parent / 'geom' / 'h8.xyz'
with geom_path.open('r') as f:
    xyz = f.read()

# ── 2. Molecule + RHF ──────────────────────────────────────────────────────
mol = gto.M(atom=xyz, basis='sto-3g', output='h8_sto3g.log', verbose=0)
mf = scf.RHF(mol).run()
print("RHF energy =", mf.e_tot)

# Reference CASCI
cas = mcscf.CASCI(mf, sum(ncas_f), sum(nelecas_f)).run()
print("CASCI energy =", cas.e_tot)

# ── 3. LASSCF ──────────────────────────────────────────────────────────────
las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f, verbose=3)
mo_loc = las.localize_init_guess(frag_atom_list, mf.mo_coeff)
las.kernel(mo_loc)
print("LASSCF energy =", las.e_tot)

# ── 5. Excitation selection (all, to be filtered by fraction) ──────────────
_, g_sel, a_idxs_all, i_idxs_all = grad.get_grad_exact(las, epsilon=0.0)
g_all = np.array(g_sel)[:, 0]

# ── 6. LSI_LUSCC (sweep over fractions) ────────────────────────────────────
for frac in FRACS:
    a_idxs, i_idxs, _ = get_sorted_excitations(
        a_idxs_all, i_idxs_all, g_all, fraction=frac)
    print(f"\nFraction: {frac} | Number of excitations: {len(a_idxs)}")
    lsi_luscc = LSI_LUSCC(las, a_idxs, i_idxs)
    e_roots, si = lsi_luscc.kernel()
    print("LSI-LUSCC ground state energy =", e_roots[0])
