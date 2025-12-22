import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi
from mrh.my_pyscf.tools import molden
from c2h4n4_struct import structure as struct
from pathlib import Path
from mrh.exploratory.citools import grad, lasci_ominus1
from mrh.exploratory.unitary_cc import lasuccsd
import time
from itertools import product
from helper import util
from lcc.lcc_solver import FCISolver_CC
import ast
import sys
pwd = Path(__file__).resolve().parent
VERBOSE = 1
# Using LASSI[r,q]

r = 1
q = 2
frac = 0.04
ncas_f = (3,3)
nelecas_f = ((2,1),(1,2))
nelecas = tuple(sum(x) for x in zip(*nelecas_f))
dnn0 = float(sys.argv[1])
dnn1 = float(sys.argv[2])
# dnn0 = 2.0
# dnn1 = 2.0

# grad_path = pwd / 'data' / 'lcc_grad_cas6r1q2_dr22.txt'
grad_path = pwd / 'data' / f'lcc_grad_cas6r1q2_dnn{dnn0}_{dnn1}.txt'
mol = struct (dnn0, dnn1, '6-31g')
mol.output = pwd / 'data' / 'c2h4n4_lassirq_631g.log'
mol.verbose = lib.logger.INFO
mol.build ()
mf = scf.RHF (mol).run ()

las = LASSCF (mf, ncas_f, nelecas_f)
las = las.state_average ([0.5,0.5],
    spins=[[1,-1],[-1,1]],
    smults=[[2,2],[2,2]],    
    charges=[[0,0],[0,0]])
mo = las.sort_mo ([16,18,22,23,24,26])
mo = las.localize_init_guess ((list (range (5)), list (range (5,10))), mo)
las.kernel (mo)
molden.from_lasscf (las, pwd / 'data' / 'c2h4n4_lasscf66_631g.molden')

# mc = mcscf.CASCI (mf, 6, 6).set (fcisolver=csf_solver(mol,smult=1))
mc = mcscf.CASCI (mf, 6, 6)
mc.kernel (las.mo_coeff)
molden.from_mcscf (mc, pwd / 'data' / 'c2h4n4_casscf66_631g.molden', cas_natorb=True)

print ("LASSCF((3,3),(3,3)) energy =", las.e_tot)
print ("CASCI(6,6) energy =", mc.e_tot)
