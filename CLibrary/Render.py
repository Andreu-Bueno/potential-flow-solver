import matplotlib.pyplot as plt

from .DataModels import *
from matplotlib.patches import Polygon
from scipy.ndimage import gaussian_filter


def PlotGrid(grid: Grid,
             geometry,
             field: np.ndarray,
             u: np.ndarray, v: np.ndarray,
             V_inf: float = 120,
             field_name: str = "psi",
             contrast: str = 'viridis',
             contourn: bool = False,
             streamplot: bool = False,
             percentile: bool = False):
    # ________________Local variables______________
    x_coordinates = np.linspace(0, grid.x_Length, grid.X_steps)
    y_coordinates = np.linspace(0, grid.y_Length, grid.Y_steps)
    X, Y = np.meshgrid(x_coordinates, y_coordinates, indexing='xy')

    color_geometry: str = 'white'
    # _____________________________________________

    # ________________Plotting______________
    plt.figure(figsize=(10, 8))

    # Configurar campo escalar
    if field_name == "psi":
        color_label = "Stream Function (Ψ)"
        clim_min, clim_max = 0, 4 * V_inf
        stream_field = field
        stream_levels = np.linspace(stream_field.min(), stream_field.max(), 30) if not np.isclose(stream_field.max(),
                                                                                                  stream_field.min()) else \
            [stream_field.min()]

        # Dibujar el campo de velocidades con flechas (quiver)
        skip = (slice(None, None, 17), slice(None, None, 17))  # para que no haya demasiadas flechas
        if not streamplot:
            plt.quiver(X[skip], Y[skip], u[skip], v[skip], color='cyan', scale=V_inf * 35)

    elif field_name == "pressure":
        color_label = "Pressure (Pa)"
        if percentile:
            clim_min, clim_max = np.percentile(field, [1, 99])
        else:
            clim_min, clim_max = field.min(), field.max()
        # Para las streamlines, recomputamos psi usando u, v si se quisiera (opcional)
        stream_field = None  # No usar streamlines directamente desde presión
        stream_levels = None
    elif field_name == "velocity":
        color_label = "Velocity (m/s)"
        field = np.sqrt(u**2 + v**2)
        if percentile:
            clim_min, clim_max = np.percentile(field, [1, 99])
        else:
            clim_min, clim_max = field.min(), field.max()
        stream_field = None
        stream_levels = None
    else:
        raise ValueError("field_name debe ser 'psi' o 'pressure'")

    # Mapa de colores
    field_smooth = gaussian_filter(field, sigma=1)
    if not streamplot:
        plt.imshow(field_smooth, extent=[0, grid.x_Length, 0, grid.y_Length],
                   origin='lower', aspect='equal', cmap=contrast)

    if field_name == "pressure":
        # Añadir contornos para resaltar el campo
        contrast = 'jet'
        plt.contourf(X, Y, field_smooth, levels=40, cmap=contrast, alpha=1)
    if not streamplot:
        plt.colorbar(label=color_label)
        plt.clim(clim_min, clim_max)
    if contourn:
        contour_levels = np.linspace(clim_min, clim_max, 40)
        plt.contour(X, Y, field_smooth, levels=contour_levels, colors='black', linewidths=0.3, alpha=0.6)

    # Dibujar líneas de corriente si se puede
    if streamplot:
        field_name = "Streamlines"
        speed = np.sqrt(u ** 2 + v ** 2)
        stream = plt.streamplot(x_coordinates, y_coordinates, u, v,
                                color=speed, cmap=contrast,
                                density=1.5,
                                norm=plt.Normalize(vmin=speed.min(), vmax=speed.max()),
                                integration_direction='both')
        plt.colorbar(stream.lines, label="Velocity (m/s)")
        color_geometry = 'gray'

    elif stream_field is not None and stream_levels is not None:
        plt.contour(X, Y, stream_field, levels=stream_levels, colors='white', linewidths=0.5)

    # Dibujar el cilindro
    if hasattr(geometry, 'Type') and geometry.Type == "cylinder":
        circle = plt.Circle((geometry.x_center, geometry.y_center), geometry.R,
                            color=color_geometry, linewidth=1, zorder=10)
        plt.gca().add_patch(circle)

    elif hasattr(geometry, 'coords_transformed'):  # entonces es un Airfoil
        airfoil_polygon = Polygon(geometry.coords_transformed, closed=True,
                                  color=color_geometry, zorder=10, linewidth=1)
        plt.gca().add_patch(airfoil_polygon)

    plt.xlim(0, grid.x_Length)
    plt.ylim(0, grid.y_Length)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"Field: {field_name.capitalize()}", size=15)
    plt.tight_layout()
    plt.show()


def PlotGrid2(grid: Grid, geometry, u: np.ndarray, v: np.ndarray,
              field: np.ndarray = None,
              V_inf: float = 120,
              field_name: str = "psi",
              contrast: str = 'viridis',
              contourn: bool = False):
    import matplotlib.pyplot as plt
    import numpy as np

    x_coordinates = np.linspace(0, grid.x_Length, grid.X_steps)
    y_coordinates = np.linspace(0, grid.y_Length, grid.Y_steps)
    X, Y = np.meshgrid(x_coordinates, y_coordinates, indexing='xy')

    plt.figure(figsize=(10, 8))

    # Configurar campo escalar
    if field_name == "psi":
        color_label = "Stream Function (Ψ)"
        clim_min, clim_max = 0, 4 * V_inf
        stream_field = field
        stream_levels = np.linspace(stream_field.min(), stream_field.max(), 70) \
            if not np.isclose(stream_field.max(), stream_field.min()) else [stream_field.min()]
        skip = (slice(None, None, 17), slice(None, None, 17))
        plt.quiver(X[skip], Y[skip], u[skip], v[skip], color='cyan', scale=V_inf * 35)

    elif field_name == "pressure":
        color_label = "Pressure (Pa)"
        clim_min, clim_max = field.min(), field.max()
        #clim_min, clim_max = np.percentile(field, 0.5), np.percentile(field, 95)
        stream_field = None
        stream_levels = None
    elif field_name == "velocity":
        color_label = "Velocity (m/s)"
        clim_min, clim_max = field.min(), field.max()
        stream_field = None
        stream_levels = None
    else:
        raise ValueError("field_name debe ser 'psi' o 'pressure'")

    # Mapa de colores
    field_smooth = gaussian_filter(field, sigma=2)
    plt.imshow(field_smooth, extent=[0, grid.x_Length, 0, grid.y_Length],
               origin='lower', aspect='equal', cmap=contrast)
    if field_name == "pressure":
        # Añadir contornos para resaltar el campo
        plt.contourf(X, Y, field_smooth, levels=40, cmap=contrast, alpha=1)
    plt.colorbar(label=color_label)
    plt.clim(clim_min, clim_max)
    if contourn:
        contour_levels = np.linspace(clim_min, clim_max, 40)
        plt.contour(X, Y, field_smooth, levels=contour_levels, colors='black', linewidths=0.3, alpha=0.6)

    # Contornos
    if stream_field is not None and stream_levels is not None:
        plt.contour(X, Y, stream_field, levels=stream_levels, colors='white', linewidths=0.5)

    # Dibuja geometría
    if hasattr(geometry, 'Type') and geometry.Type == "cylinder":
        circle = plt.Circle((geometry.x_center, geometry.y_center), geometry.R,
                            color='white', linewidth=1, zorder=10)
        plt.gca().add_patch(circle)

    elif hasattr(geometry, 'coords_transformed'):  # entonces es un Airfoil
        airfoil_polygon = Polygon(geometry.coords_transformed, closed=True,
                                  color='white', zorder=10, linewidth=1)
        plt.gca().add_patch(airfoil_polygon)

    # Plot configs
    plt.xlim(0, grid.x_Length)
    plt.ylim(0, grid.y_Length)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"Field: {field_name.capitalize()}")
    plt.tight_layout()
    plt.show()
