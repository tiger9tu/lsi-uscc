import numpy as np
import matplotlib.pyplot as plt

# ── Data ──────────────────────────────────────────────────────────────────
fracs   = [0.10, 0.20, 0.30, 0.40]
e_luscc = [-295.4886339642, -295.4945146817, -295.5005542707, -295.5087806143]
n_exc   = [668, 1340, 2006, 2676]
n_states_luscc = [n * 2 for n in n_exc]

e_las    = -295.46390407
e_lassis = -295.52016414967636
e_casci  = -295.52422953

n_states_lassis = 321
n_states_casci  = 19404

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(fracs, e_luscc, 'o-', color='steelblue', linewidth=2,
        markersize=7, label='LSI-LUSCC')

# Annotate each LSI-LUSCC point with number of states
for f, e, ns in zip(fracs, e_luscc, n_states_luscc):
    ax.annotate(f'{ns}', xy=(f, e), xytext=(0, 8),
                textcoords='offset points', ha='center', fontsize=9, color='steelblue')

ax.axhline(e_las,    color='gray',   linestyle='--', linewidth=1.5, label=f'LAS ({e_las:.5f})')
ax.axhline(e_lassis, color='orange', linestyle='--', linewidth=1.5, label=f'LASSIS ({e_lassis:.5f})')
ax.axhline(e_casci,  color='red',    linestyle='--', linewidth=1.5, label=f'CASCI ({e_casci:.5f})')

ax.text(0.97, e_lassis, f'{n_states_lassis}', color='orange', fontsize=9,
        va='bottom', ha='right')
ax.text(0.97, e_casci,  f'{n_states_casci}',  color='red',    fontsize=9,
        va='bottom', ha='right')

ax.set_xlabel('Fraction of excitations', fontsize=13)
ax.set_ylabel('Energy (Hartree)', fontsize=13)
ax.set_title('LSI-LUSCC energy vs excitation fraction\nC$_2$H$_4$N$_4$, 6-31G, CASSCF(10,10)', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(0, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('frac_sweep_plot.png', dpi=150)
print("Saved frac_sweep_plot.png")
