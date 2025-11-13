# run_tests.py

from qtest.test_base import MyNOQIStudy, MolConfig


with open('../geom/c4.xyz', 'r', encoding='utf-8') as f:
    c4xyz = f.read()

c4 = MolConfig(
    name="C4_631G",
    xyz=c4xyz,
    basis="6-31g",
    ncas=[2, 2],
    nelecas=[2, 2],
    spinsub=[1, 1],
    frag_atom_list=((0, 2), (3, 1)),
    output = "data/c4.out",
)

# Instantiate your test class
job = MyNOQIStudy(c4, AMPLITUDE=1.0, VERBOSE=4)
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
