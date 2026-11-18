"""Lift coefficient of the NACA 2412 against angle of attack, from the saved results.

The lift is computed with Solver.ComputeLiftAirfoil (pressure integrated over the surface nodes)
and compared with thin-airfoil theory, CL = 2*pi*(alpha - alpha_0), alpha_0 = -2.1 deg.

The full set of saved fields (results/Numpy/Airfoil_<angle>Deg.npz) is needed; only a few are
included in the repository, so missing angles are skipped.

Usage (from the repository root):
    python examples/lift_curve.py
"""
import contextlib
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from CLibrary import DataModels, Solver  # noqa: E402

NPZ_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DataModels.RESULTS_DIR, "Numpy")
V_INF, CHORD = 100.0, 1.5
RHO = DataModels.Atmosphere(1.225, 15, 101325, 0, 287.058).density_h  # sea level
g = DataModels.Grid(4, 4, 0.02, 0.02)
coords = os.path.join(DataModels.RESULTS_DIR, "Airfoil_txt", "NACA2412_OriginalCoordinatesTxt")

angles, cl = [], []
for alpha in range(0, 17, 2):
    path = os.path.join(NPZ_DIR, f"Airfoil_{alpha}Deg.npz")
    if not os.path.exists(path):
        continue
    pressure = np.load(path)["pressure"]
    with contextlib.redirect_stdout(io.StringIO()):
        airfoil = DataModels.Airfoil(surfaceCoordinates=coords, centreCoordinates=(2.0, 2.0), theta=alpha,
                                     chord=CHORD, kuttaCondition=True)
        lift = Solver.ComputeLiftAirfoil(g, pressure, airfoil)
    angles.append(alpha)
    cl.append(lift / (0.5 * RHO * V_INF**2 * CHORD))
    print(f"alpha = {alpha:3d} deg   L' = {lift:9.1f} N/m   CL = {cl[-1]:.3f}")

a = np.array(angles)
slope = np.polyfit(np.radians(a), cl, 1)[0]
print(f"Lift slope: {slope:.2f} per rad (thin-airfoil theory: {2 * np.pi:.2f})")

plt.figure(figsize=(6.5, 4.5))
plt.plot(a, cl, "o-", label="This solver (confined domain)")
plt.plot(a, 2 * np.pi * np.radians(a + 2.1), "k--", label=r"Thin-airfoil theory, $2\pi(\alpha-\alpha_0)$")
plt.xlabel(r"Angle of attack $\alpha$ (deg)")
plt.ylabel(r"$C_L$")
plt.title("NACA 2412 lift coefficient")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "figures", "lift_curve.png"), dpi=130)
