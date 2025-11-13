# run_tests.py

from qtest.test_base import MyNOQIStudy, MolConfig


with open('../geom/stil001.xyz', 'r', encoding='utf-8') as f:
    stil001xyz = f.read()
    
stil001 = MolConfig(
    name="stil001_631G",
    xyz=stil001xyz,
    basis="6-31g",
    ncas=[4, 2, 4],
    nelecas=[4, 2, 4],
    spinsub=[1, 1, 1],
    frag_atom_list=[ [1,2,3,4,5,6,15,16,17,18,19] , [0,7, 14,20] , [8,9,10,11,12,13, 21,22,23,24,25] ],
    output = "data/stil001.out",
)

# Instantiate your test class
job = MyNOQIStudy(stil001, AMPLITUDE=1.0, VERBOSE=4)
# Run once the SCF -> LASSCF -> CASCI stack
job.run_scf_stack_once()

# Sweep multiple excitation selections without redoing LAS/CAS
energies = job.run_selection_sweep(
    label_prefix="testscan",
    # epsilons=[0.01, 0.001],
    factors=[0.01, 0.02, 0.03, 0.04],
)

# Print or save results
for label, record in energies.items():
    print(f"\n[{label}]")
    for key, val in record.items():
        print(f"{key:>18s}: {val:.12f}")
