gpu_run=0
import pyscf
if gpu_run:
    from gpu4mrh import patch_pyscf
    from mrh.my_pyscf.gpu import libgpu
import numpy as np
from pyscf import gto, lib, scf, mcscf
from pyscf.lib import chkfile
from pyscf.tools import molden
from mrh.my_pyscf.mcscf import lasscf_async as asyn
from mrh.my_pyscf import lassi
from pyscf.mcscf import avas
if gpu_run:
    gpu=libgpu.init()
    libgpu.set_verbose_(gpu, 1)
    from pyscf.lib import param
    param.use_gpu = gpu
    param.custom_debug = False
    param.custom_fci = True
    param.mgpu_fci = True

lib.logger.TIMER_LEVEL = lib.logger.INFO

mol_name='geom.xyz' # Fe2S2 complex
basis={'C': 'cc-pvdz','H': 'cc-pvdz','S': 'aug-cc-pvdz','Fe': 'aug-cc-pvdz'}
#basis={'C': 'cc-pvdz','H': 'cc-pvdz','S': 'aug-cc-pvdz','Fe': 'aug-cc-pvdz'}
if gpu_run:
    mol=gto.M(atom=mol_name,verbose=4,spin=18,charge=-2,basis=basis, max_memory=500000, output='gpu_lassis_test_s13.log', use_gpu = gpu)
else:
    mol=gto.M(atom=mol_name,verbose=4,spin=18,charge=-2,basis=basis, max_memory=500000, output='cpu_lassis_test_s13.log') #spin = 2S
mf=scf.ROHF(mol)
mf.init_guess='atom'
mf=mf.density_fit()
mf=mf.newton()
mf.kernel()
#mf.mo_coeff=np.load('hf.npy')
ao_labels=['Fe 3d','Fe 4d']
ncas,nelecas,mo_coeff=avas.avas(mf,ao_labels,minao=mol.basis)
if gpu_run: las = asyn.LASSCF(mf, (5,5,5,5), ((5,1),(5,0),(1,5),(0,5)), spin_sub=(5,6,5,6), use_gpu = gpu, verbose=4, assert_no_dupes=False)
else: las = asyn.LASSCF(mf, (5,5,5,5), ((5,1),(5,0),(1,5),(0,5)), spin_sub=(5,6,5,6), verbose=4, assert_no_dupes=False)
atom_list = ([0],[1],[2],[3])
mo_coeff = las.set_fragments_(atom_list, mf.mo_coeff, mo_occ = mf.mo_occ)
_,_,conv_mo_coeff,_,_,_ = molden.load('converged_22e40o.molden')
mo_list = [125,126,127,128,129,135,136,137,138,139,145,146,147,148,149,155,156,157,158,159]
mo_coeff = las.sort_mo(mo_list, conv_mo_coeff) 
las.lasci_(mo_coeff)
from mrh.my_pyscf import lassi
lsi=lassi.LASSIS(las)
lsi.cisolver_attr_spin_flips['max_cycle'] = 200 # Spin flip max cycle
lsi.max_cycle_macro = 100 # Charge-hop as well as LASSI Hamiltonian diagonalization max cycle
lsi.sisolver.smult=13
lsi.sisolver.nroots = 12
lsi.sisolver.max_cycle=200
lsi.sisolver.pspace_size=400 #default seems to be 400 for si sisolver
lsi.run(opt=1)
lsi.dump_chk('s13.chk')
from mrh.my_pyscf.lassi.sitools import analyze_moments
analyze_moments(lsi, lsi.si, lsi.ci, state=range(1))
from mrh.my_pyscf import mcpdft
mc = mcpdft.LASSIS(lsi, 'tPBE', states=range(12))
mc.kernel()
exit()
for smult_si in range(17,-1,-2):
  lsi.sisolver.smult=smult_si
  lsi.sisolver.nroots = 8
  lsi.sisolver.pspace_size=600 #default seems to be 400 for si sisolver
  lsi.eig(opt=1)
  analyze_moments(lsi, lsi.si, lsi.ci, state=range(1))
  mc = mcpdft.LASSIS(lsi, 'tPBE', states=range(8))
  mc.kernel()
if gpu_run: libgpu.destroy_device(gpu)
