#!/usr/bin/env python3
"""
Show how MP2 amplitudes map onto UCC parameters.
Works for both RHF/RMP2 and UHF/UMP2 references.
"""

import numpy as np
from pyscf import gto, scf, mp
from pyscf.cc import ccsd, uccsd      # flattening helpers
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad, lasci_ominus1
# -------------------------------------------------------------
# 1.  Build a tiny test system (water, STO-3G)
# -------------------------------------------------------------
xyz =  '''    H      0.000000000000   0.000000000000   0.000000000000
    H      1.000000000000   0.000000000000   0.000000000000
    H      0.273746762116   2.195450598147   0.100000000000
    H      1.232912762116   1.895450598147  -0.100000000000
    H      0.507178110854   4.193780995243   0.049334760036
    H      1.506140937609   3.988021397347  -0.049334760036
    '''
mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log',
    verbose=0)
mf = scf.RHF (mol).run ()
mp2 = mp.MP2(mf).run()                # change to mp.UMP2(mf) for open shell

# -------------------------------------------------------------
# 2.  Grab amplitudes – supply zero singles if RMP2
# -------------------------------------------------------------
try:
    t1 = mp2.t1                       # exists only for open-shell (UMP2)
except AttributeError:
    nocc = mp2.nocc                   # occupied spatial orbitals
    nvir = mf.mo_coeff.shape[1] - nocc
    t1   = np.zeros((nocc, nvir))     # singles are rigorously zero

t2 = mp2.t2                           # RMP2: ndarray,  UMP2: (aa,ab,bb)

# -------------------------------------------------------------
# 3.  Flatten into the canonical UCC parameter vector
# -------------------------------------------------------------
if isinstance(t2, tuple):             # open-shell: three spin blocks
    ucc_vec = uccsd.amplitudes_to_vector(t1, t2)
else:                                 # closed-shell
    ucc_vec = ccsd.amplitudes_to_vector(t1, t2)

print(f"\nTotal UCC parameters: {ucc_vec.size:d}")

# -------------------------------------------------------------
# 4.  Demonstrate the mapping for doubles (closed shell case)
# -------------------------------------------------------------
nocc = mp2.nocc
nvir = t2.shape[2]
# if not isinstance(t2, tuple):
first_double = nocc * nvir        # singles occupy the first slice

print("mp2 energy : {:.12f} Eh".format(mp2.e_tot))

print("\nIndex :  |  Excitation  |  Amplitude")
print("-----------------------------------------------")
idx = first_double
for i in range(nocc):
    for j in range(i+1, nocc):
        for a in range(nvir):
            for b in range(a+1, nvir):
                amp = ucc_vec[idx]
                print(f"{idx:5d} : | {i:1d},{j:1d} → {a+nocc},{b+nocc} |  {amp:+.6e}")
                idx += 1


def get_x_amplitudes(a_idxs, i_idxs, amplitudes, init_amplitudes):
    """Set the x amplitudes in the FCISolver's psi object. 
    The init amplitudes are MP2 amplitudes, which is for all double
    excitations, but we only set for selected ones in a_idxs and i_idxs."""
    x = np.zeros(len(a_idxs))
    idx = None
    for i, (a, i_) in enumerate(zip(a_idxs, i_idxs)):
        if len(a) < 2:
            continue # There is only double excitations for init_amplitudes
        x[i] = init_amplitudes[a[0],i[0],a[1],i[1]] # The order
    return x


a_idxs = []
i_idxs = []

for i in range(nocc):
    for j in range(i+1, nocc):
        for a in range(nvir):
            for b in range(a+1, nvir):
                a_idxs.append((a + nocc, b + nocc))
                i_idxs.append((i, j))

mc_uscc = mcscf.CASCI(mf, 6, 6)
mc_uscc.mo_coeff = mf.mo_coeff

lasci_ominus1.GLOBAL_MAX_CYCLE = 15000

print("a_idxs = ", a_idxs)
print("i_idxs = ", i_idxs)

aidx_formated = [np.array(item, dtype=np.uint8) for item in a_idxs]
iidx_formated = [np.array(item, dtype=np.uint8) for item in i_idxs]
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, aidx_formated, iidx_formated)
mc_uscc.fcisolver.norb_f = [2, 2, 2]  # (2e, 2o) excitations
mc_uscc.fcisolver.frozen = "CI"
mc_uscc.kernel()

psi = mc_uscc.fcisolver.psi

print("optimized amplitudes  =", psi.x)
print("mc_uscc.e_tot = ", mc_uscc.e_tot)

h1eff,e_core= mc_uscc.get_h1eff(mc_uscc.mo_coeff)
h2eff = mc_uscc.get_h2eff()
h = [e_core, h1eff, h2eff]

psi_eng = psi.energy_tot(psi.x, h)
print("psi energy = ", psi_eng)


psi.x[1] = -0.06232327058193741  # Set the first x amplitude to a mp2 value

print("setting psi.x[1] to MP2 value = ", psi.x[1])
print("psi energy after setting = ", psi.energy_tot(psi.x, h))
