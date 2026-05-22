import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Data ────────────────────────────────────────────────────────────────────────
fracs   = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
n_exc   = [  77,  155,  232,  310,  387,  464,  542,  619,  697,  774]
e_luscc = [-295.62973607, -295.63283916, -295.63464967, -295.64300819,
           -295.64710652, -295.64726005, -295.64726005, -295.64726005,
           -295.64726005, -295.64726005]

e_casci  = -295.64726956
e_lassis = -295.63181536
e_lasscf = -295.53609835453517

# convert to mH relative to CASCI
def to_mH(e):
    return (np.array(e) - e_casci) * 1000

de_luscc  = to_mH(e_luscc)
de_lassis = to_mH([e_lassis])[0]
de_lasscf = to_mH([e_lasscf])[0]

# ── Plot ────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

# LSI-LUSCC curve
ax.plot(fracs, de_luscc, 'o-', color='steelblue', lw=2, ms=6,
        label='LSI-LUSCC (m=2)', zorder=3)

# CASCI reference
ax.axhline(0.0, color='black', lw=1.5, ls='--', label=f'CASCI(6,6)  (0 mH ref)')

# LASSIS
ax.axhline(de_lassis, color='tomato', lw=1.5, ls='--',
           label=f'LASSIS  ({de_lassis:+.2f} mH)')

# LASSCF (far above — shown as arrow annotation instead of full line)
ax.annotate(f'LASSCF  ({de_lasscf:+.1f} mH)',
            xy=(0.15, de_lassis + 1.5), fontsize=8.5, color='gray',
            ha='left')

# chemical accuracy band
ax.axhspan(-1.594, 1.594, alpha=0.12, color='green', label='Chemical accuracy (±1.594 mH)')

# number of states for each LSI-LUSCC point (n_exc * 2)
n_states = [n * 2 for n in n_exc]

# label n_states directly on each point of the LSI-LUSCC curve
for f, de, ns in zip(fracs, de_luscc, n_states):
    ax.annotate(str(ns), xy=(f, de), xytext=(0, 7),
                textcoords='offset points', ha='center', fontsize=7,
                color='steelblue', fontweight='bold')

# label state counts on the left of the reference lines
ax.text(0.06, de_lassis, '88', va='center', ha='left', fontsize=7,
        color='tomato', fontweight='bold', transform=ax.get_yaxis_transform())
ax.text(0.06, 0.0, '400', va='bottom', ha='left', fontsize=7,
        color='black', fontweight='bold', transform=ax.get_yaxis_transform())

ax.set_xlabel('Fraction of excitations', fontsize=12)
ax.set_ylabel('ΔE from CASCI (mH)', fontsize=12)
ax.set_title('LSI-LUSCC energy vs. excitation fraction\nC₂H₄N₄ / 6-31g,  m=2,  CASCI(6,6) reference', fontsize=11)
ax.set_xlim(0.05, 1.05)
ax.set_xticks(fracs)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, which='major', ls=':', alpha=0.5)

plt.tight_layout()
plt.savefig('chn6_frac_sweep.png', dpi=150)
print("Saved chn6_frac_sweep.png")
