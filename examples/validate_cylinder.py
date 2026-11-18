"""Compare the pressure coefficient around the fixed cylinder with the analytical potential-flow solution.

Cp is sampled on a circle of radius r = 1.2 R, just outside the stair-stepped cylinder boundary,
and compared with the exact unbounded-domain solution at the same radius:

    |V|^2 / V_inf^2 = (1 - a^2)^2 cos^2(theta) + (1 + a^2)^2 sin^2(theta),  a = R / r
    Cp = 1 - |V|^2 / V_inf^2

Usage (from the repository root):
    python examples/validate_cylinder.py
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

V_INF, R, XC, YC = 100.0, 0.5, 2.0, 2.0
R_SAMPLE = 1.2 * R

g = DataModels.Grid(4, 4, 0.02, 0.02)
x = np.linspace(0, g.x_Length, g.X_steps)
y = np.linspace(0, g.y_Length, g.Y_steps)
cyl = DataModels.Cylinder(XC, YC, R, 0, "cylinder")
with contextlib.redirect_stdout(io.StringIO()):  # the solver prints the full arrays
    psi = Solver.SetBoundaryConditions(g, x, y, cyl, V_inf=V_INF)
    psi = Solver.SolveStreamFunction(psi, g.delta_X, g.delta_Y, g.X_steps, g.Y_steps, cyl,
                                     max_iterations=5000, omega=1.8)
    u, v = Solver.ComputeVelocityField(psi, g)
speed = np.sqrt(u**2 + v**2)  # rows are y, columns are x


def bilinear(field, xq, yq):
    fi, fj = yq / g.delta_Y, xq / g.delta_X
    i0, j0 = int(np.floor(fi)), int(np.floor(fj))
    ti, tj = fi - i0, fj - j0
    return ((1 - ti) * (1 - tj) * field[i0, j0] + ti * (1 - tj) * field[i0 + 1, j0]
            + (1 - ti) * tj * field[i0, j0 + 1] + ti * tj * field[i0 + 1, j0 + 1])


theta = np.linspace(0, np.pi, 91)  # measured from the upstream stagnation point
xs, ys = XC - R_SAMPLE * np.cos(theta), YC + R_SAMPLE * np.sin(theta)
cp_num = 1 - np.array([bilinear(speed, a, b) for a, b in zip(xs, ys)])**2 / V_INF**2
a2 = (R / R_SAMPLE)**2
cp_exact = 1 - ((1 - a2)**2 * np.cos(theta)**2 + (1 + a2)**2 * np.sin(theta)**2)
print(f"Cp at theta = 90 deg: numerical {cp_num[45]:.3f}, analytical {cp_exact[45]:.3f}")

plt.figure(figsize=(7, 4.5))
plt.plot(np.degrees(theta), cp_exact, "k-", label="Analytical (unbounded domain)")
plt.plot(np.degrees(theta), cp_num, "o", ms=4, mfc="none", label="Numerical (this solver)")
plt.xlabel(r"$\theta$ from the free-stream direction (deg)")
plt.ylabel(r"$C_p$")
plt.title(r"Pressure coefficient around the cylinder at $r = 1.2R$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "figures", "validation_cp.png"), dpi=130)
