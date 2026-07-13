"""Single-panel schematic plots for c6, stilbene-90 singlet, chn_10_10 at FRAC=0.02.

Loads the already-computed *_frac0.02.npz files in ./data/ and renders one
3D schematic per system with:
  - true (non-exaggerated) LAS / LUSCC / USCC / P_USCC vectors
  - anisotropic e2/e3 axis scaling so small angles are visible
  - tip markers (screen-pixel circles) instead of quiver arrowheads
  - inline state labels with a white bbox so they read over the tick numbers
  - an energy panel anchored to the axes
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path


PWD  = Path(__file__).resolve().parent
DATA = PWD / "data"

JOBS = [
    "c6_frac0.02",
    "stil90_singlet_frac0.02",
    "chn_10_10_frac0.02",
]

LABEL_BBOX = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85)
ENERGY_BBOX = dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9)


def make_plot(stem: str) -> Path:
    d = np.load(DATA / f"{stem}.npz")
    LAS  = np.array([1.0, 0.0, 0.0])
    LUS  = d["LUSCC_xyz"]
    USC  = d["USCC_xyz"]
    PUSC = d["PUSCC_xyz"]
    e_las  = float(d["e_las"])
    e_uscc = float(d["e_uscc"])
    e_lus  = float(d["e_luscc"])

    # Stretch e2 / e3 so small components fill the panel; cap at the e1 scale.
    y_max = min(max(LUS[1], USC[1]) * 1.5 + 0.02, 1.1)
    z_max = min(max(USC[2], 1e-3)    * 1.5 + 0.02, 1.1)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    O = np.zeros(3)
    for color, v in [("k", LAS), ("tab:blue", LUS),
                     ("tab:red", USC), ("tab:orange", PUSC)]:
        ax.plot([O[0], v[0]], [O[1], v[1]], [O[2], v[2]],
                color=color, linewidth=2.0)
        ax.scatter([v[0]], [v[1]], [v[2]],
                   color=color, s=45, depthshade=False, zorder=5)
    plane = np.array([O, LAS * 1.05, LUS * 1.05])
    ax.plot_trisurf(plane[:, 0], plane[:, 1], plane[:, 2],
                    alpha=0.10, color="tab:blue")
    ax.plot(*zip(PUSC, USC), color="tab:red",    ls=":", lw=1.3)
    ax.plot(*zip(PUSC, LUS), color="tab:orange", ls=":", lw=1.3)

    ax.text(LAS[0]  + 0.02, LAS[1],                LAS[2],
            "|LAS>",        color="k",
            fontsize=10, bbox=LABEL_BBOX, zorder=6)
    ax.text(LUS[0]  - 0.05, LUS[1]  + 0.04 * y_max, LUS[2],
            "|LAS_LUSCC>",  color="tab:blue",
            fontsize=10, bbox=LABEL_BBOX, zorder=6)
    ax.text(USC[0]  - 0.10, USC[1]  + 0.02 * y_max, USC[2] + 0.05 * z_max,
            "|LAS_USCC>",   color="tab:red",
            fontsize=10, bbox=LABEL_BBOX, zorder=6)
    ax.text(PUSC[0] + 0.01, PUSC[1] - 0.10 * y_max, PUSC[2],
            "P|LAS_USCC>",  color="tab:orange",
            fontsize=10, bbox=LABEL_BBOX, zorder=6)

    ax.set_xlim(0, 1.1); ax.set_ylim(0, y_max); ax.set_zlim(0, z_max)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

    numbers = (
        f"E(USCC)  = {e_uscc:.6f}\n"
        f"E(LUSCC) = {e_lus:.6f}\n"
        f"E(LAS)   = {e_las:.6f}"
    )
    ax.text2D(0.02, 0.98, numbers, transform=ax.transAxes,
              ha="left", va="top", family="monospace", fontsize=9,
              bbox=ENERGY_BBOX)

    fig.tight_layout()
    out = DATA / f"{stem}_v2.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    for stem in JOBS:
        out = make_plot(stem)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
