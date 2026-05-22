"""Check how many pair evaluations occur in root_make_rdm3s for lsi_prime with m=2."""
import numpy as np
from pyscf import gto, scf, lib
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.my_pyscf import lassi
from mrh.exploratory.citools.grad import get_grad_exact_lassi
import mrh.my_pyscf.lassi.op_o0 as op_o0
from lcc import LSI_LUSCC

mol = gto.M(atom='H 0 0 0; H 0 0 1.5; H 0 0 3; H 0 0 4.5',
            basis='sto-3g', verbose=0)
mf = scf.RHF(mol).run()

las = LASSCF(mf, (2, 2), (2, 2), spin_sub=(1, 1))
mo = las.localize_init_guess([[0, 1], [2, 3]])
las.kernel(mo)

lsi = lassi.LASSIS(las)
lsi.kernel()
print(f"Full LASSIS nroots: {len(lsi.si[0])}")
print(f"Top |ci| coefficients: {np.sort(np.abs(lsi.si[:, 0]))[::-1][:5]}")

# Patch _make_rdm3s_spinless_pair to count calls
call_count = [0]
_orig = op_o0._make_rdm3s_spinless_pair
def _patched(*args, **kwargs):
    call_count[0] += 1
    return _orig(*args, **kwargs)
op_o0._make_rdm3s_spinless_pair = _patched

for m in [1, 2, 4]:
    call_count[0] = 0
    lsi_prime = LSI_LUSCC(lsi, [], [], top_m=m)
    lsi_prime.kernel()
    call_count[0] = 0  # reset after kernel (which may also call rdm3s internally)
    get_grad_exact_lassi(lsi_prime, state=0)
    print(f"m={m}: _make_rdm3s_spinless_pair called {call_count[0]} times (expected {m**2})")

op_o0._make_rdm3s_spinless_pair = _orig
