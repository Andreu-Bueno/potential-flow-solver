from CLibrary import DataModels
from CLibrary import Solver
from CLibrary import Render
from CLibrary import StreamWriter
from CLibrary import StreamReader
import numpy as np
import os

airfoil = DataModels.Airfoil(
    surfaceCoordinates=os.path.join(DataModels.RESULTS_DIR, "Airfoil_txt", "NACA2412_OriginalCoordinatesTxt"),
    centreCoordinates=(2.0, 2.0),
    theta=10,
    chord=1.5,
    kuttaCondition=True)

airfoil.plot_airfoil()

lengths: [int, int] = [4, 4]
g = DataModels.Grid(lengths[0], lengths[1], 0.02, 0.02)
atmosphere = DataModels.Atmosphere(1.225, 15, 101325, 10000, 287.058)
x_coords = np.linspace(0, g.x_Length, g.X_steps)
y_coords = np.linspace(0, g.y_Length, g.Y_steps)


# The parameters that can be changed are the next:
# Grid: (dimensions of space, step)
#
# Atmosphere: (only need to change altitude and will already have the computed data in the class as attributes)
#
# Airfoil: (can set any airfoil coordinates from Airfoil Tools database in
#           results/Airfoil_txt/...,
#           change AoA by setting theta=..., or whether)
#           to set kuttaCondition as true or false)
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
    # Executes the numerical process for the simulation of a cylinder (if want to execute it go further bellow and
    # write in main(execute=) the number 0)
    if execute == 0:
        psi = Solver.SetBoundaryConditions(g, x_coords, y_coords, airfoil, V_inf=100)
        psi = Solver.SolveStreamFunction(psi, g.delta_X, g.delta_Y, g.X_steps, g.Y_steps, airfoil,
                                         max_iterations=5000, omega=1.8)

        u, v = Solver.ComputeVelocityField(psi, g)
        print(" ============================================================= ")
        print("~~VELOCITY FIELD~~")
        print(u, v)

        pressure = Solver.ComputePressureDistribution(u, v, atmosphere, V_inf=100)

        lift = Solver.ComputeLiftAirfoil(g, pressure, airfoil)
        print(f"Lift force: {lift}")

        # StreamWriter.SaveToNPZ(
        #    "Airfoil_5Deg_10000m.npz",
        #    psi, u, v, pressure)

        Render.PlotGrid(g, airfoil, psi, u, v, V_inf=100)
        Render.PlotGrid(g, airfoil, pressure, u, v, V_inf=100, field_name="pressure", contourn=True)

    # In case of not wanting to go through numerical computation you can already compile some examples from the folder
    # Results\Numpy\... and just change the last part of the path (name) by setting execute=1 further bellow.
    if execute == 1:
        psi, u, v, pressure = StreamReader.LoadFromNPZ(
            os.path.join("Numpy", "Airfoil_10Deg.npz"))

        if psi is not None:
            print("Datos cargados correctamente.")
            print(Solver.ComputeLiftAirfoil(g, pressure, airfoil, seePoints=False))

        Render.PlotGrid(g, airfoil, psi, u, v, V_inf=100)
        Render.PlotGrid(g, airfoil, pressure, u, v, V_inf=100, field_name="pressure", contourn=True)
        Render.PlotGrid(g, airfoil, psi, u, v, V_inf=100, field_name="velocity", contourn=True, contrast="jet")
        Render.PlotGrid(g, airfoil, pressure, u, v, V_inf=100, streamplot=True, contrast='jet')


if __name__ == '__main__':
    main(execute=1)
