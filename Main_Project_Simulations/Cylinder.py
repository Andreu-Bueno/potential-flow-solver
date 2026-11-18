import math
import os

import matplotlib.pyplot as plt

from CLibrary import DataModels
from CLibrary import Solver
from CLibrary import StreamReader
from CLibrary import Render
from CLibrary import StreamWriter


# The parameters that can be changed are the next:
# Grid: (dimensions of space, step)
#
# Atmosphere: (only need to change altitude and will already have the computed data in the class as attributes)
#
# Cylinder: (position of the center of the circumference -> (x, y), radius, rotation speed (not based in real physics))
#
# Func -> SetBoundaryConditions: needs the parameters np.linspace for x and y
#
# Func -> (cand enter either a cylinder or an airfoil, can define with max_iterations the maximum number of iterations
#          to stop if convergence is not achieved, w(omega - SOR method step) (preferably do not change it, it is already adjusted))
#
# Func -> ComputeLift: In case you want to see the surface selected you can set the parameter seeCoordinates as True
#
# Func -> RenderPlot: in case of plotting another thing that's not psi, enter the data (np.ndarray) and set the
#         parameter field_name as pressure or velocity.
#
#         Furthermore, in case of just wanting to plot the streamlines just instantiate ->
#         -> Render.PlotGrid(g, cylinder, psi, u, v, V_inf=120, streamplot=True). Ca
#
#         In case of you want better differentiation of colormap can set parameter contourn as True. Furthermore,
#         you can also change the contrast with contrast='...', and percentil=True.


def main(execute: int = 0):
    import numpy as np

    if execute == 0:
        # Executes the numerical process for the simulation of a cylinder (if want to execute it go further bellow and
        # write in main(execute=) the number 0)
        lengths: [int, int] = [4, 4]
        g = DataModels.Grid(lengths[0], lengths[-1], 0.02, 0.02)
        atmosphere = DataModels.Atmosphere(1.225, 15, 101325, 0, 287.058)
        x_coords = np.linspace(0, g.x_Length, g.X_steps)
        y_coords = np.linspace(0, g.y_Length, g.Y_steps)

        cylinder = DataModels.Cylinder(lengths[0] / 2, lengths[0] / 2, 0.5, 0, "cylinder")

        psi = Solver.SetBoundaryConditions(g, x_coords, y_coords, cylinder, V_inf=100)
        psi = Solver.SolveStreamFunction(psi, g.delta_X, g.delta_Y, g.X_steps, g.Y_steps, cylinder,
                                         max_iterations=5000, omega=1.8)

        u, v = Solver.ComputeVelocityField(psi, g)
        print(" ============================================================= ")
        print("~~VELOCITY FIELD~~")
        print(u, v)

        pressure = Solver.ComputePressureDistribution(u, v, atmosphere)

        lift = Solver.ComputeLift(g, pressure, cylinder)
        print(f"Lift force: {lift}")

        Render.PlotGrid(g, cylinder, psi, u, v, V_inf=120)
        Render.PlotGrid(g, cylinder, pressure, u, v, V_inf=120, field_name="pressure")

        cylinder = DataModels.Cylinder(lengths[0] / 2, lengths[0] / 2, 0.5, 35, "cylinder")

        psi = Solver.SetBoundaryConditions(g, x_coords, y_coords, cylinder, V_inf=120)
        psi = Solver.SolveStreamFunction(psi, g.delta_X, g.delta_Y, g.X_steps, g.Y_steps, cylinder,
                                         max_iterations=5000, omega=1.8)
        u, v = Solver.ComputeVelocityField(psi, g)
        print(" ============================================================= ")
        print("~~VELOCITY FIELD~~")
        print(u, v)

        pressure = Solver.ComputePressureDistribution(u, v, atmosphere)

        lift = Solver.ComputeLift(g, pressure, cylinder)
        print(f"Lift force: {lift}")

        Render.PlotGrid(g, cylinder, psi, u, v, V_inf=120)
        Render.PlotGrid(g, cylinder, pressure, u, v, V_inf=120, field_name="pressure")

        cylinder = DataModels.Cylinder(lengths[0] / 2, lengths[0] / 2, 0.5, -35, "cylinder")

        psi = Solver.SetBoundaryConditions(g, x_coords, y_coords, cylinder, V_inf=120)
        psi = Solver.SolveStreamFunction(psi, g.delta_X, g.delta_Y, g.X_steps, g.Y_steps, cylinder,
                                         max_iterations=5000, omega=1.8)
        u, v = Solver.ComputeVelocityField(psi, g)
        print(" ============================================================= ")
        print("~~VELOCITY FIELD~~")
        print(u, v)

        pressure = Solver.ComputePressureDistribution(u, v, atmosphere)

        lift = Solver.ComputeLift(g, pressure, cylinder)
        print(f"Lift force: {lift}")

        Render.PlotGrid(g, cylinder, psi, u, v, V_inf=120)
        Render.PlotGrid(g, cylinder, pressure, u, v, V_inf=120, field_name="pressure")

    # In case of not wanting to go through numerical computation you can already compile some examples from the folder
    # Results\Numpy\Cylinder and just change the last part of the path (name) by setting execute=1 further bellow.
    elif execute == 1:
        psi, u, v, pressure = StreamReader.LoadFromNPZ(
            os.path.join("Numpy", "Cylinder", "Clockwise_w=50_Rotation.npz"))

        if psi is not None:
            print("Datos cargados correctamente.")
            g = DataModels.Grid(4, 4, 0.02, 0.02)
            cylinder = DataModels.Cylinder(2, 2, 0.5, 100, "cylinder")

            print(Solver.ComputeLift(g, pressure, cylinder))
            Render.PlotGrid(g, cylinder, psi, u, v, V_inf=100)
            Render.PlotGrid(g, cylinder, pressure, u, v, V_inf=100, field_name="pressure", contourn=True)
            Render.PlotGrid(g, cylinder, psi, u, v, V_inf=100, field_name="velocity", contourn=True)
            Render.PlotGrid(g, cylinder, psi, u, v, V_inf=100, streamplot=True)


if __name__ == '__main__':
    main(execute=1)
