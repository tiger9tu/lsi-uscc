import numpy as np
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.exploratory.citools import grad
from mrh.my_pyscf import lassi
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from lcc import LSI_LUSCC
from helper.util import get_sorted_excitations
from pathlib import Path
from time import time

# ── 1. System parameters ───────────────────────────────────────────────────
FRAC = 0.02  # fraction of excitations to include; set to e.g. 0.02 for smaller run

ncas_f         = (4, 2, 4)
nelecas_f      = ((2, 2), (1, 1), (2, 2))
spin_sub_f     = (1, 1, 1)
frag_atom_list = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]

# ── 2. Molecule + RHF ──────────────────────────────────────────────────────
lib.logger.TIMER_LEVEL = lib.logger.INFO
mol = struct(2.0, 2.0, '6-31g')
mol.output = 'c2h4n4_631g.log'
mol.verbose = 3
mol.spin = 8
mol.build()
mf = scf.RHF(mol).run()

ncas = sum(ncas_f)
nelecas = (5, 5)

# Reference CASCI
mc = mcscf.CASCI(mf, ncas, nelecas).set(fcisolver=csf_solver(mol, smult=1))
mc.kernel()
print("CASCI(10,10) energy =", mc.e_tot)

# ── 3. LASSCF ──────────────────────────────────────────────────────────────
las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f)
mo_coeff = las.localize_init_guess(frag_atom_list)
las.kernel(mo_coeff)
print("LASSCF energy =", las.e_tot)

# ── 4. LASSIS ──────────────────────────────────────────────────────────────
lsi = lassi.LASSIS(las)
t0 = time()
e_lsi, si = lsi.kernel()
print("LASSIS energy =", e_lsi[0])
print("LASSIS time: {:.2f} s".format(time() - t0))
print("nroots =", len(e_lsi))

# ── 5. Excitation selection ────────────────────────────────────────────────
_, g_sel, a_idxs_all, i_idxs_all = grad.get_grad_exact(las, epsilon=0.0)
g_all = np.array(g_sel)[:, 0]
a_idxs, i_idxs, _ = get_sorted_excitations(
    a_idxs_all, i_idxs_all, g_all, fraction=FRAC)
print(f"Number of excitations selected (fraction={FRAC}): {len(a_idxs)}")

# ── 6. LSI_LUSCC ────────────────────────────────────────────────────────────
t0 = time()
lsi_luscc = LSI_LUSCC(las, a_idxs, i_idxs)
e_roots, si_rq = lsi_luscc.kernel()
print("LSI-LUSCC ground state energy =", e_roots[0])
print("Time: {:.2f} s".format(time() - t0))
