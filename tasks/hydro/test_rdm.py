import numpy as np
import pyscf
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad, lasci_ominus1
from lcc.lcc_solver import FCISolver_CC
from mrh.my_pyscf import lassi
from helper.util import print_list_matrix, get_sorted_excitations, cilas2f, lassi_rdm2_to_lasscf

from pathlib import Path



pwd = Path(__file__).resolve().parent
geom_path = pwd.parent / 'geom' / 'h4.xyz'
with geom_path.open('r') as f:
    xyz = f.read()

# Initializing the molecule with RHF
#===================================
ncas_f = (2,2)
nelecas_f = (2,2)
spin_sub_f = (1,1)
frag_atom_list = ((0,1),(2,3))

mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log',
    verbose=0)
mf = scf.RHF (mol).run ()
# ref = mcscf.CASCI (mf, 4, 4).run () # = FCI
print ("RHF energy = ", mf.e_tot)
# print ("CASCI energy = ", ref.e_tot)


# Running LASSCF
#===================================
las = LASSCF (mf, ncas_f, nelecas_f, spin_sub=spin_sub_f)
las.verbose = 4

mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)
print ("LASSCF energy = ", las.e_tot)

rdm1s = las.make_casdm1s ()
rdm2s = las.make_casdm2s ()
print("las rdm1s.shape =", rdm1s.shape)
print("las rdm2s.shape =", rdm2s.shape)


las2 = las.state_average ([1,0],
    # spins=[[1,-1],[-1,1]],
    smults=[[1,1],[1,3]],
    charges=[[0,0],[0,0]])

las2.lasci ()
las2.dump_spaces ()
lsi = lassi.LASSI(las2)
e_roots, si_hand = lsi.kernel()
print ("LASSI(hand) energy =", e_roots[0])
print ("SI vector (hand):")
print (si_hand[:,0])

lsirdm1s, lsirdm2s = lsi.make_casdm12s ()
print("lassi rdm1s.shape =", lsirdm1s.shape)
print("lassi rdm2s.shape =", lsirdm2s.shape)

lsirdm1s0 = np.array(lsirdm1s[0])
shouldbe0 = lsirdm1s0 - rdm1s
print("Difference in RDM1s between LASSCF and LASSI (should be 0):")
print("norm of difference: ", np.linalg.norm(shouldbe0))

lsirdm2s0 = lassi_rdm2_to_lasscf(lsirdm2s[0])
shouldbe2 = lsirdm2s0 - rdm2s
print("Difference in RDM2s between LASSCF and LASSI (should be 0):")
print("norm of difference: ", np.linalg.norm(shouldbe2))



