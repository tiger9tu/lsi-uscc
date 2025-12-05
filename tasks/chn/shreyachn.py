import sys
import copy
import numpy as np
import h5py
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.my_pyscf.tools import molden
from mrh.tests.lasscf.c2h4n4_struct import structure as struct
from lasvqe.las_vqe import LASVQE
from qiskit_qulacs.qulacs_estimator import QulacsEstimator

lib.logger.TIMER_LEVEL = lib.logger.INFO
mol = struct (2.0, 2.0, '6-31g')
mol.output = 'c2h4n4_631g_las3_rscan.log'
mol.verbose = lib.logger.INFO
mol.spin = 8
mol.build ()
mf = scf.RHF (mol).run ()

las = LASSCF (mf, (4,2,4), ((2,2),(1,1),(2,2)), spin_sub=(1,1,1))
mo_coeff = las.localize_init_guess ([[0,1,2],[3,4,5,6],[7,8,9]])
las.kernel (mo_coeff)
mo_coeff = las.mo_coeff

mc = mcscf.CASCI (mf, 10, (5,5)).set (fcisolver=csf_solver(mol,smult=1))
mc.kernel (mo_coeff)

sys.stderr.flush ()
print ("LASSCF((4,4),(2,2),(4,4)) energy =", las.e_tot)
print ("CASCI(10,10) energy =", mc.e_tot, flush=True)

fmt_str = 'ROW: {:.1f} {:s} {:.9f} {:s} {:.9f}'
e_lasscf = las.e_states[0]
print (fmt_str.format (2.0, str(las.converged), e_lasscf, str(mc.converged), mc.e_tot))

def compute_lasuscc(las, eps):

    lasvqe = LASVQE(mf, las, f_orbs=(4,2,4), f_elec=(4,2,4), f_atom_list=[[0,1,2],[3,4,5,6],[7,8,9]],  spin_sub=(1,1,1), selected=True, epsilon=eps)
    vqe_en, vqe_result = lasvqe.run(estimator=QulacsEstimator(), gate_counts=True, verbose=3)
	print(f"LAS-USCC-VQE Energy: {vqe_en:.12f} Ha | Epsilon: {eps:.12f}")

    return

# Epsilons =  [0.0140047  0.00700483 0.0070026  0.00466867 0.00420959 0.00240517 0.00233465 0.00228203 0.00172649 0.00138026 0.00109903 0.00079211 0.00074442 0.00052791 0.00026381]


lsi = las.as_scanner ()
for dr in np.arange (3.5, -0.31, -0.1):
    mol1 = struct (dr, dr, '6-31g')
    e_lasscf = lsi (mol1)
    mc = mcscf.CASCI (lsi._scf, 10, (5,5)).set (fcisolver = csf_solver (mol, smult=1))
    mc.max_cycle = 100
    mc.kernel (lsi.mo_coeff)
    e_casci = mc.e_tot
    compute_lasuscc(lsi, 0.0140047)
    print (fmt_str.format (dr, str(lsi.converged), e_lasscf, str(mc.converged), e_casci))