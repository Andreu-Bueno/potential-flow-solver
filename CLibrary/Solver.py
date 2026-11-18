import numpy
from .DataModels import *
from matplotlib.path import Path


def SetBoundaryConditions(grid: Grid, x_coordinates, y_coordinates,
                          geometry, V_inf: float = 100) -> np.ndarray:
    psi = np.zeros((grid.X_steps, grid.Y_steps))
    for i in range(grid.X_steps):
        psi[i, 0] = V_inf * y_coordinates[i]
        psi[i, -1] = psi[i, 0]
    for j in range(grid.Y_steps):
        psi[0, j] = 0
        psi[-1, j] = x_coordinates[-1] * V_inf
    print(" ============================================================= ")
    print("~~PSI BOUNDARY CONDITIONS~~")
    print(psi)

    X, Y = np.meshgrid(x_coordinates, y_coordinates, indexing='xy')

    if hasattr(geometry, 'Type') and geometry.Type == "cylinder":
        collision = (X - geometry.x_center) ** 2 + (Y - geometry.y_center) ** 2 <= geometry.R ** 2
        V_rot = geometry.w * geometry.R
        psi_cylinder = V_inf * geometry.y_center - 5 * V_rot
        for i in range(grid.X_steps):
            for j in range(grid.Y_steps):
                if collision[i, j]:
                    psi[i, j] = psi_cylinder
        print(" ============================================================= ")
        print("~~PSI~~")
        print(psi)

    elif hasattr(geometry, 'coords_transformed'):
        geometry: Airfoil
        # Marcamos puntos dentro del perfil
        airfoil_path = Path(geometry.coords_transformed)
        points = np.vstack((X.ravel(), Y.ravel())).T
        mask_inside = airfoil_path.contains_points(points).reshape(X.shape)

        if geometry.kuttaCondition:
            # flujo uniforme en centro de cuerda
            psi_body = V_inf * (geometry.trailing_edge[1] - numpy.sin(geometry.theta_rad))
            psi[mask_inside] = psi_body
        elif not geometry.kuttaCondition and geometry.theta != 0:
            # desplazamiento vertical para romper la condición de Kutta
            delta_y = 0.2 * geometry.chord
            psi_body = V_inf * (geometry.trailing_edge[1] + delta_y)
        elif not geometry.kuttaCondition and geometry.theta == 0:
            psi_body = V_inf * geometry.centreCoordinates[0]

        from shapely.geometry import Point, Polygon, LineString
        # Paso adicional: marcar la psi en los nodos que están sobre el borde del perfil
        airfoil_polygon = Polygon(geometry.coords_transformed)

        for i in range(grid.X_steps):
            for j in range(grid.Y_steps):
                point = Point(X[i, j], Y[i, j])
                distance = airfoil_polygon.exterior.distance(point)
                if distance < min(grid.delta_X, grid.delta_Y):  # cercano al contorno
                    psi[i, j] = psi_body
    return psi


def SolveStreamFunction(psi, delta_x: float, delta_y: float, x_steps: int, y_steps: int, geometry,
                        max_iterations: int = 1000, omega: float = 1.7, tolerance: float = 1e-6) -> np.ndarray:
    beta = (delta_x ** 2) / (delta_y ** 2)
    x_coords = np.linspace(0, delta_x * (x_steps - 1), x_steps)
    y_coords = np.linspace(0, delta_y * (y_steps - 1), y_steps)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')

    if hasattr(geometry, 'Type') and geometry.Type == "cylinder":
        collision = (X - geometry.x_center) ** 2 + (Y - geometry.y_center) ** 2 <= geometry.R ** 2
    elif hasattr(geometry, 'coords_transformed'):
        # Detectar si un punto está dentro del perfil alar
        points = np.vstack((X.ravel(), Y.ravel())).T
        airfoil_path = Path(geometry.coords_transformed)
        collision = airfoil_path.contains_points(points).reshape(X.shape)
    else:
        raise ValueError("Geometría no soportada")

    for iteration in range(max_iterations):
        max_diff = 0.0
        for i in range(1, x_steps - 1):
            for j in range(1, y_steps - 1):
                if not collision[i, j]:
                    psi_old = psi[i, j]
                    psi_new_val = (psi[i + 1, j] + psi[i - 1, j] +
                                   beta * (psi[i, j + 1] + psi[i, j - 1])) / (2 * (1 + beta))
                    psi[i, j] = (1 - omega) * psi_old + omega * psi_new_val
                    max_diff = max(max_diff, abs(psi[i, j] - psi_old))
        if iteration % 100 == 0 or max_diff < tolerance:
            print(f"Iteración {iteration}, cambio máximo: {max_diff:.2e}")
        if max_diff < tolerance:
            print("¡Convergencia alcanzada!")
            break
    print(" ============================================================= ")
    print("~~PSI SOLVED~~")
    print(psi)
    return psi


def ComputeVelocityField(psi: np.ndarray, grid: Grid,  V_inf: float = None, geometry=None, delta_x: float = None, delta_y: float = None,
                         x_steps: int = None, y_steps: int = None) -> tuple:
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)

    rows, columns = psi.shape  # obtener las filas y columnas de la grid de psi

    for i in range(rows):
        for j in range(columns):
            if 1 <= i < rows - 1:
                u[i, j] = (psi[i + 1, j] - psi[i - 1, j]) / (2 * grid.delta_Y)
            elif i == 0:
                u[i, j] = (psi[i + 1, j] - psi[i, j]) / grid.delta_Y
            else:  # i == rows - 1
                u[i, j] = (psi[i, j] - psi[i - 1, j]) / grid.delta_Y

            if 1 <= j < columns - 1:
                v[i, j] = -(psi[i, j + 1] - psi[i, j - 1]) / (2 * grid.delta_X)
            elif j == 0:
                v[i, j] = -(psi[i, j + 1] - psi[i, j]) / grid.delta_X
            else:  # j == cols - 1
                v[i, j] = -(psi[i, j] - psi[i, j - 1]) / grid.delta_X

    if geometry is not None:
        x_coords = np.linspace(0, delta_x * (x_steps - 1), x_steps)
        y_coords = np.linspace(0, delta_y * (y_steps - 1), y_steps)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
        points = np.vstack((X.ravel(), Y.ravel())).T
        airfoil_path = Path(geometry.coords_transformed)
        collision = airfoil_path.contains_points(points).reshape(X.shape)
        u[collision] = 0
        v[collision] = 0
    return u, v


def ComputePressureDistribution(u: np.ndarray, v: np.ndarray, atmosphere: Atmosphere, V_inf: float = 100, geometry=None,
                                delta_x: float = None, delta_y: float = None, x_steps: int = None, y_steps: int = None) -> np.ndarray:
    V_local = np.sqrt(u ** 2 + v ** 2)

    # Detectar si un punto está dentro del perfil alar
    if geometry is not None:
        x_coords = np.linspace(0, delta_x * (x_steps - 1), x_steps)
        y_coords = np.linspace(0, delta_y * (y_steps - 1), y_steps)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
        points = np.vstack((X.ravel(), Y.ravel())).T
        airfoil_path = Path(geometry.coords_transformed)
        collision = airfoil_path.contains_points(points).reshape(X.shape)
        V_local[collision] = 0

    pressure = atmosphere.pressure_h + 0.5 * atmosphere.density_h * (V_inf ** 2 - V_local ** 2)
    print(" ============================================================= ")
    print("~~PRESSURE~~")
    print(pressure)
    return pressure


def ComputeNormals(cylinder: Cylinder, delta_x: float, x_steps: int,
                   delta_y: float, y_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_coords = np.linspace(0, delta_x * (x_steps - 1), x_steps)
    y_coords = np.linspace(0, delta_y * (y_steps - 1), y_steps)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')

    # Calculate exact distance to cylinder surface
    dist = np.sqrt((X - cylinder.x_center) ** 2 + (Y - cylinder.y_center) ** 2) - cylinder.R

    # Find the closest points to the surface
    epsilon = min(delta_x, delta_y) * 0.6  # About half grid spacing
    on_surface = np.abs(dist) <= epsilon

    # Initialize normals
    normal_x = np.zeros_like(X)
    normal_y = np.zeros_like(Y)

    # Calculate dx/dy from center
    dx = X - cylinder.x_center
    dy = Y - cylinder.y_center

    # Assign simplified normals ensuring symmetry
    for i in range(x_steps):
        for j in range(y_steps):
            if on_surface[i, j]:
                if abs(dx[i, j]) > abs(dy[i, j]):
                    normal_x[i, j] = 1 if dx[i, j] > 0 else -1
                else:
                    normal_y[i, j] = 1 if dy[i, j] > 0 else -1

    return normal_x, normal_y, on_surface


def ComputeNormalsAirfoil(geometry, delta_x: float, x_steps: int,
                          delta_y: float, y_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_coords = np.linspace(0, delta_x * (x_steps - 1), x_steps)
    y_coords = np.linspace(0, delta_y * (y_steps - 1), y_steps)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')

    normal_x = np.zeros_like(X)
    normal_y = np.zeros_like(Y)

    if hasattr(geometry, "Type") and geometry.Type == "cylinder":
        dist = np.sqrt((X - geometry.x_center) ** 2 + (Y - geometry.y_center) ** 2) - geometry.R
        epsilon = min(delta_x, delta_y) * 0.6
        on_surface = np.abs(dist) <= epsilon

        dx = X - geometry.x_center
        dy = Y - geometry.y_center
        for i in range(x_steps):
            for j in range(y_steps):
                if on_surface[i, j]:
                    r = np.sqrt(dx[i, j] ** 2 + dy[i, j] ** 2)
                    normal_x[i, j] = dx[i, j] / r
                    normal_y[i, j] = dy[i, j] / r

    elif hasattr(geometry, "coords_transformed"):
        # Perfil alar
        points = np.vstack((X.ravel(), Y.ravel())).T
        path = Path(geometry.coords_transformed)
        inside = path.contains_points(points).reshape(X.shape)

        # Superficie = borde cercano al perfil
        from scipy.ndimage import binary_dilation
        on_surface = binary_dilation(inside) & (~inside)

        # Estimar normales por gradiente numérico
        grad_y, grad_x = np.gradient(inside.astype(float))
        mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8
        normal_x = grad_x / mag
        normal_y = grad_y / mag
        normal_x[~on_surface] = 0
        normal_y[~on_surface] = 0

    else:
        raise ValueError("Geometría no soportada")

    return normal_x, normal_y, on_surface


def ComputeLiftAirfoil(grid: Grid, pressure: np.ndarray, geometry, seePoints: bool = False) -> float:
    # Get normals and surface points
    normal_x, normal_y, on_surface = ComputeNormalsAirfoil(geometry, grid.delta_X, grid.X_steps,
                                                           grid.delta_Y, grid.Y_steps)

    # Visual verification (optional)
    if seePoints:  # Set to True to debug
        plt.scatter(*np.where(on_surface), c=normal_y[on_surface])
        plt.title("Surface points and their normal_y components")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    # Calculate lift
    p_surface = pressure[on_surface]
    ny_surface = normal_y[on_surface]

    # Segment length - use delta_x for vertical segments, delta_y for horizontal
    is_horizontal = np.abs(normal_x[on_surface]) > 0
    delta_s = np.where(is_horizontal, grid.delta_Y, grid.delta_X)

    lift: float = np.sum(p_surface * ny_surface * delta_s)
    return lift


def ComputeLift(grid: Grid, pressure: np.ndarray, geometry, seeCoordinates: bool = False) -> float:
    # Get normals and surface points
    normal_x, normal_y, on_surface = ComputeNormals(geometry, grid.delta_X, grid.X_steps,
                                                    grid.delta_Y, grid.Y_steps)

    # Visual verification (optional)
    if seeCoordinates:  # Set to True to debug
        plt.scatter(*np.where(on_surface), c=normal_y[on_surface])
        plt.title("Surface points and their normal_y components")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    # Calculate lift
    p_surface = pressure[on_surface]
    ny_surface = normal_y[on_surface]

    # Segment length - use delta_x for vertical segments, delta_y for horizontal
    is_horizontal = np.abs(normal_x[on_surface]) > 0
    delta_s = np.where(is_horizontal, grid.delta_Y, grid.delta_X)

    lift: float = -np.sum(p_surface * ny_surface * delta_s)
    return lift
