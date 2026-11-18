"""Render the README figures from the saved results in results/Numpy using CLibrary.Render.

Usage (from the repository root):
    python examples/make_figures.py
"""
import contextlib
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from CLibrary import DataModels, Render, StreamReader  # noqa: E402

FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
_next_name = []
plt.show = lambda *a, **k: (plt.savefig(os.path.join(FIG_DIR, _next_name.pop(0)), dpi=110), plt.close("all"))

g = DataModels.Grid(4, 4, 0.02, 0.02)
coords = os.path.join(DataModels.RESULTS_DIR, "Airfoil_txt", "NACA2412_OriginalCoordinatesTxt")

with contextlib.redirect_stdout(io.StringIO()):
    airfoil = DataModels.Airfoil(surfaceCoordinates=coords, centreCoordinates=(2.0, 2.0), theta=10, chord=1.5,
                                 kuttaCondition=True)
psi, u, v, p = StreamReader.LoadFromNPZ(os.path.join("Numpy", "Airfoil_10Deg.npz"))
_next_name += ["airfoil_10deg_streamfunction.png", "airfoil_10deg_pressure.png", "airfoil_10deg_streamlines.png"]
Render.PlotGrid(g, airfoil, psi, u, v, V_inf=100)
Render.PlotGrid(g, airfoil, p, u, v, V_inf=100, field_name="pressure", contourn=True)
Render.PlotGrid(g, airfoil, p, u, v, V_inf=100, streamplot=True, contrast="jet")

for name, w, out in [("ResultsNoRotation.npz", 0, "cylinder_streamfunction.png"),
                     ("Clockwise_w=50_Rotation.npz", 50, "rotating_cylinder_streamfunction.png")]:
    psi, u, v, p = StreamReader.LoadFromNPZ(os.path.join("Numpy", "Cylinder", name))
    cyl = DataModels.Cylinder(2, 2, 0.5, w, "cylinder")
    _next_name += [out]
    Render.PlotGrid(g, cyl, psi, u, v, V_inf=100)
print("Figures saved to", FIG_DIR)
