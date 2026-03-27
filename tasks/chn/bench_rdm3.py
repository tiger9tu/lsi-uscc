"""Benchmark: wall time for root_make_rdm3s on CHN LASSIS, m=1..8.

|lsi_m> = (1/sqrt(m)) * sum_{k=0}^{m-1} |las_k>
"""
import sys, numpy as np
from time import time
from pyscf import gto, scf, lib
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.my_pyscf import lassi
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from mrh.my_pyscf.lassi.op_o0 import root_make_rdm3s

def p(msg=''):
    print(msg, flush=True)

# ── System setup ──────────────────────────────────────────────────────────────
ncas_f         = (4, 2, 4)
nelecas_f      = ((2, 2), (1, 1), (2, 2))
spin_sub_f     = (1, 1, 1)
frag_atom_list = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]

mol = struct(2.0, 2.0, '6-31g')
mol.verbose = 0
mol.spin    = 8
mol.max_memory = 16000
mol.build()
mf = scf.RHF(mol).run()
p(f"RHF energy:    {mf.e_tot:.10f}")

las = LASSCF(mf, ncas_f, nelecas_f, spin_sub=spin_sub_f, verbose=0)
mo  = las.localize_init_guess(frag_atom_list)
las.kernel(mo)
p(f"LASSCF energy: {las.e_tot:.10f}")

# ── LASSIS ────────────────────────────────────────────────────────────────────
p("\nRunning LASSIS ...")
lsi = lassi.LASSIS(las)
lsi.verbose = 0
t0 = time()
e_roots, si = lsi.kernel()
t_lassi = time() - t0
nprods, nroots_si = si.shape
p(f"LASSIS:  {t_lassi:.1f}s   nprods={nprods}   nroots={nroots_si}")
p(f"Energies (first 8): {e_roots[:min(8, nroots_si)]}")

ci_fr     = lsi.ci
nelec_frs = lsi.get_nelec_frs()
norb      = lsi.ncas
norb_f    = lsi.ncas_sub

p(f"\nnorb={norb}   norb_f={norb_f}")
p()
p(f"{'m':>3}  {'nnz_coeff':>10}  {'nnz_pairs':>10}  {'wall(s)':>10}  "
  f"{'aaa_max':>12}  {'aab_max':>12}  {'abb_max':>12}  {'bbb_max':>12}")
p("-" * 100)

for m in range(1, 9):
    n = min(m, nroots_si)
    w = 1.0 / np.sqrt(float(n))
    si_art = np.zeros((nprods, 1))
    si_art[:, 0] = w * np.sum(si[:, :n], axis=1)

    coeffs = si_art[:, 0]
    nnz = int(np.sum(np.abs(coeffs) > 1e-14))

    t0 = time()
    rdm3s = root_make_rdm3s(lsi, ci_fr, nelec_frs, si_art, ix=0)
    t_rdm3 = time() - t0

    maxs = [float(np.max(np.abs(rdm3s[s]))) for s in range(4)]
    p(f"  {m:>1}  {nnz:>10}  {nnz**2:>10}  {t_rdm3:>10.2f}  "
      f"  {maxs[0]:>10.4f}    {maxs[1]:>10.4f}    {maxs[2]:>10.4f}    {maxs[3]:>10.4f}")
