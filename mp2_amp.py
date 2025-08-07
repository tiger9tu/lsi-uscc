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
xyz = '''H 0.0 0.0 0.0;
            H 1.0 0.0 0.0;
            H 0.2 1.6 0.1;
            H 1.159166 1.3 -0.1'''
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
if not isinstance(t2, tuple):
    first_double = nocc * nvir        # singles occupy the first slice

    print("\nIndex :  |  Excitation  |  Amplitude")
    print("-----------------------------------------------")
    idx = first_double
    for i in range(nocc):
        for j in range(i+1, nocc):
            for a in range(nvir):
                for b in range(a+1, nvir):
                    amp = ucc_vec[idx]
                    print("t2[{:d},{:d},{:d},{:d}] = {}".format(i, j, a, b, amp))
                    print(f"{idx:5d} : | {i:1d},{j:1d} → {a+nocc},{b+nocc} |  {amp:+.6e}")
                    idx += 1


def set_x_amplitudes(a_idxs, i_idxs, amplitudes, init_amplitudes):
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
                a_idxs.append((a, b))
                i_idxs.append((i, j))

mc_uscc = mcscf.CASCI(mf, 4, 4)
mc_uscc.mo_coeff = mf.mo_coeff

lasci_ominus1.GLOBAL_MAX_CYCLE = 0
mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, a_idxs_selected, i_idxs_selected)

mc_uscc.kernel()
print("mc_uscc.e_tot = ", mc_uscc.e_tot)


