import numpy as np
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.citools import grad
from mrh.my_pyscf.lassi import LASSI
from lcc import LSI_LUSCC
from helper.util import get_sorted_excitations
from pathlib import Path

# ── 1. System parameters ───────────────────────────────────────────────────
FRAC = 0.1

ncas_f      = (2, 2)
nelecas_f   = (2, 2)
spin_sub_f  = (1, 1)
frag_atom_list = ((0, 1), (2, 3))

pwd = Path(__file__).resolve().parent
geom_path = pwd.parent / 'geom' / 'h4.xyz'
with geom_path.open('r') as f:
    xyz = f.read()

# ── 2. Molecule + RHF ──────────────────────────────────────────────────────
mol = gto.M(atom=xyz, basis='sto-3g', output='h4_sto3g.log',
            verbose=lib.logger.DEBUG4)
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

# ── 5. Excitation selection ────────────────────────────────────────────────
_, g_sel, a_idxs, i_idxs = grad.get_grad_exact(las, epsilon=0.0)
g = np.array(g_sel)[:, 0]
a_idxs, i_idxs, _ = get_sorted_excitations(a_idxs, i_idxs, g, fraction=FRAC)

# ── 6. LSI_LUSCC ────────────────────────────────────────────────────────────
lsi_luscc = LSI_LUSCC(las, a_idxs, i_idxs)
e_roots, si = lsi_luscc.kernel()

# ── 7. Output ──────────────────────────────────────────────────────────────
print("LSI-LUSCC ground state energy =", e_roots[0])
