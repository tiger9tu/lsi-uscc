# run_tests.py

from qtest.test_base import MyNOQIStudy, MolConfig


import os
pwd = os.path.dirname(os.path.abspath(__file__))
with open(f'{pwd}/../geom/h4.xyz', 'r', encoding='utf-8') as f:
    h4xyz = f.read()


h4 = MolConfig(
    name="H4_STO3G",
    xyz=h4xyz,
    basis="sto-3g",
    ncas=[2, 2],
    nelecas=[2, 2],
    spinsub=[1, 1],
    frag_atom_list=((0, 1), (2, 3)),
    output = f"{pwd}/data/h4.out",
)

# Instantiate your test class
job = MyNOQIStudy(h4, AMPLITUDE=1000, VERBOSE=4)

# Run once the SCF -> LASSCF -> CASCI stack
job.run_scf_stack_once()

# Sweep multiple excitation selections without redoing LAS/CAS
energies = job.run_selection_sweep(
    label_prefix="testscan",
    epsilons=[0.02],
    # epsilons=[0.01, 0.001],
    # factors=[0.1],
)

# Print or save results
for label, record in energies.items():
    print(f"\n[{label}]")
    for key, val in record.items():
        print(f"{key:>18s}: {val:.12f}")
