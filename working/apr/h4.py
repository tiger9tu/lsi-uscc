import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi
from mrh.my_pyscf.tools import molden
from c2h4n4_struct import structure as struct

# Using a hand-made model space

xyz = '''H 0.0 0.0 0.0;
            H 1.0 0.0 0.0;
            H 0.2 1.6 0.1;
            H 1.159166 1.3 -0.1'''
mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log',
    verbose=0)
mol.output = 'c2h4n4_lassi_631g.log'
mol.verbose = lib.logger.INFO
mol.build ()
mf = scf.RHF (mol).run ()

# las = LASSCF (mf, (3,3), ((2,1),(1,2)))
las = LASSCF (mf, (2,2), ((1,1),(1,1)))
# las = las.state_average ([0.5,0.5],
#     spins=[[1,-1],[-1,1]],
#     smults=[[2,2],[2,2]],    
#     charges=[[0,0],[0,0]])


# mo = las.sort_mo ([16,18,22,23,24,26])
# mo = las.localize_init_guess ((list (range (5)), list (range (5,10))), mo)
frag_atom_list = ((0,1),(2,3))
mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)
print("lasci = ", las.ci)

ref = mcscf.CASSCF (mf, 4, 4).run () 
print("las energy = ", las.e_tot)
print("casscf energy =", ref.e_tot)

# Use LASSIrq for automatic CT state generation
r = 1
q = 1
lsi = lassi.LASSIrq(las, r=r, q=q)
e_roots, si_rq = lsi.kernel()

print ("LASSI[{},{}] energy =".format(r, q), e_roots[0])
print("All e_roots = ", e_roots)
print ("SI vector (LASSI[{},{}]):".format(r, q))
print (si_rq[:,0])

molden.from_lassi (lsi, 'h4_lassirq_sto3g.molden', state=0)

