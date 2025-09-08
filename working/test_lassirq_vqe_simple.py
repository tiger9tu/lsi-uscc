#!/usr/bin/env python3
"""
Simple test of LASSIrq-VQE on H4 molecule
"""

from pyscf import gto, scf, lib
from lassi_vqe_class import LASSI_VQE

def test_h4_lassirq_vqe():
    """Test LASSIrq-VQE on H4 molecule"""
    
    # H4 geometry
    h4_xyz = '''H 0.0 0.0 0.0
    H 1.0 0.0 0.0
    H 0.273746762116 2.195450598147 0.100000000000
    H 1.232912762116 1.895450598147 -0.100000000000'''
    
    def mol_func(charge, spin, basis):
        return gto.M(
            atom=h4_xyz,
            basis=basis,
            charge=charge,
            spin=spin,
            verbose=0
        )
    
    try:
        # Create LASSI_VQE instance
        lassi_vqe = LASSI_VQE(
            mol=mol_func,
            ncas_sub=(2, 2),
            nelec_sub=((2, 0), (0, 2)),  # Specify as tuples of (nalpha, nbeta)
            basis='sto-3g',
            output_file=None,
            verbose=0,
            r=1, q=1
        )
        
        # Run calculation
        final_energies, final_eigenvectors, lassirq_energies = lassi_vqe.run_full_calculation(
            frag_atom_list=((0, 1), (2, 3)),
            vqe_max_cycle=2,
            vqe_conv_tol=1e-4
        )
        
        print("LASSIrq-VQE test completed successfully!")
        print(f"LASSIrq-VQE ground state energy: {final_energies[0]:.10f} hartree")
        print(f"LASSIrq ground state energy: {lassirq_energies[0]:.10f} hartree")
        print(f"Energy difference (VQE - LASSIrq): {final_energies[0] - lassirq_energies[0]:+.10f} hartree")
        
        return True
        
    except Exception as e:
        print(f"LASSIrq-VQE test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_h4_lassirq_vqe()
    if success:
        print("\n✓ LASSIrq-VQE implementation is working!")
    else:
        print("\n✗ LASSIrq-VQE needs debugging")