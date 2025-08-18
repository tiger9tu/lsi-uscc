"""
Example: reproduce MP2 energy with a CASCI object fed the MP2 T2 amplitudes
Tested with PySCF ≥ 2.4
"""

import numpy as np
from pyscf import gto, scf, mp, mcscf, ci

# ----------------------------------------------------------------------
# 1. Molecule and RHF
# ----------------------------------------------------------------------
mol = gto.Mole()
mol.atom  = "H 0 0 0;  F 0 0 0.9"      # small, fast example
mol.basis = "sto-3g"
mol.spin  = 0                          # closed shell
mol.build()

mf = scf.RHF(mol).run()

# ----------------------------------------------------------------------
# 2. MP2 – grab correlation energy and T2 amplitudes
# ----------------------------------------------------------------------
mp2 = mp.MP2(mf).run()                 # mp2.e_tot = HF + corr
t2  = mp2.t2                           # shape (nocc,nocc,nvir,nvir) :contentReference[oaicite:0]{index=0}
print("MP2 total energy  : {:.12f} Eh".format(mp2.e_tot))

# ----------------------------------------------------------------------
# 3. Build a CI vector containing (HF + all T2) only
# ----------------------------------------------------------------------
nocc = t2.shape[0]
nvir = t2.shape[2]
c0   = 1.0                             # coefficient of HF reference
c1   = np.zeros((nocc, nvir))          # no singles (MP2 uses canonical HF MOs)
c2   = t2.copy()

ci_vec = ci.cisd.amplitudes_to_cisdvec(c0, c1, c2)  # :contentReference[oaicite:1]{index=1}

# ----------------------------------------------------------------------
# 4. CASCI driver that *uses CISD* as the solver
#    (active space = all orbitals, so the CI solver sees exactly the HF
#     determinant, singles, and doubles)
# ----------------------------------------------------------------------
ncas     = mol.nao_nr()                # all orbitals in the active space
nelecas  = mol.nelectron
# --- everything until ci_vec stays exactly the same ------------------------

# build the CASCI driver with RCISD as its solver
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver = ci.cisd.RCISD(mf)

# -------------------------------------------------------
# instead of   eci = mc.fcisolver.energy(eris, ci_vec)
# use .kernel() once with max_cycle = 0   (zero Davidson iterations)
# -------------------------------------------------------
eris = mc.fcisolver.ao2mo()       # 2-e integrals in the CAS
mc.fcisolver.max_cycle = 0        # ask for *one* H·C evaluation
_, eci, _ = mc.fcisolver.kernel(eris, ci0=ci_vec)   # conv flag, E, ci_out

e_tot = eci + mc.get_ecore()
print(f"CASCI(HF+T2) energy : {e_tot: .12f} Eh")

