import numpy as np                                                                                                                  
import sys 
from pyscf import gto, scf, lib 
from pyscf import gto, scf, tools, mcscf
from pyscf.mcscf import avas
from pyscf.mcscf import project_init_guess
from pyscf import mcscf
from mrh.my_pyscf.tools import molden
from mrh.my_pyscf.mcscf import lasscf_async as asyn
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf.lasscf_rdm2 import LASSCF as LASSCF_rdm
#from mrh.my_pyscf.fci import csf_solver
mol_name='Fe_Fe_Fe.xyz' # Al-Fe-Fe MOF node
#basis0={'C': 'sto-3g','H': 'sto-3g','O': 'sto-3g','Al': 'cc-pvdz','Fe': 'cc-pvdz'}
basis={'C': 'cc-pvdz','H': 'cc-pvdz','O': 'cc-pvtz','Al': 'cc-pvtz','Fe': 'cc-pvtz'}
output='lasscf_hf.log'
#mol0=gto.M(atom=mol_name,verbose=4,spin=14,charge=0,basis=basis0, output=output) #spin = 2S
mol=gto.M(atom=mol_name,verbose=4,spin=10,charge=0,basis=basis, output=output) #spin = 2S
mol.max_memory = 256000
mf=scf.ROHF(mol)
mf.init_guess='atom'
mf=mf.density_fit()
mf.max_cycle=1
mf.kernel()
mf=mf.newton()
mf.max_cycle=1
mf.kernel()
mf.mo_coeff = np.load('hf_is.npy')
exit()
#np.save('hf_is.npy',mf.mo_coeff)
#mf.mo_coeff=np.load('hf.npy')
ncas,nelecas,guess_mo_coeff=avas.kernel(mf,['Fe 3d', 'Fe 4d'],minao=mol.basis, openshell_option=3)
mc_test = mcscf.CASCI (mf, ncas, nelecas)
tools.molden.from_mo(mol,'avas.molden',guess_mo_coeff)
 
final_list=[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]+mc_test.ncore
las=LASSCF(mf,(5,5,5),((5,1),(4,1),(4,1)),spin_sub=(5,4,4),verbose=4) # spin is 2S+1
las_rdm = LASSCF_rdm(mf,(5,5,5),((5,1),(4,1),(4,1)),spin_sub=(5,4,4),verbose=4)
mo_sorted=las.sort_mo(final_list,guess_mo_coeff)#,final_list)
#mo_coeff=mf.mo_coeff
#mo_coeff[:,mc_test:mc_test+ncas]=mo_coeff
#mo_coeff=np.load('las_14_14.npy')
mo_localized=las.localize_init_guess(([17],[19],[22]),mo_sorted)#, freeze_cas_spaces=False)
mo_localized_rdm=las_rdm.localize_init_guess(([17],[19],[22]),mo_sorted)
las.max_cycle_macro=200
las_rdm.max_cycle_marco = 200 
las.kernel(mo_localized)
las_rdm.kernel(mo_localized_rdm)
np.save('las_15_16_is',las.mo_coeff)
np.save('las_15_16_rdm_is',las_rdm.mo_coeff)
molden.from_lasscf(las, 'las_15_16_is.molden')#,cas_natorb=True)
molden.from_lasscf(las_rdm, 'las_15_16_is_rdm.molden')
