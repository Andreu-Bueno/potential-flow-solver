from typing import Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os

# Folder with the airfoil coordinates and the saved results (relative to the repository root)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


@dataclass
class Grid:
    x_Length: float
    y_Length: float
    delta_X: float
    delta_Y: float

    def __post_init__(self):
        assert self.delta_X > 0 and self.delta_Y > 0, "Delta values must be positive."
        assert self.x_Length == self.y_Length or self.y_Length == self.x_Length

        self.X_steps = round(self.x_Length / self.delta_X) + 1
        self.Y_steps = round(self.y_Length / self.delta_Y) + 1


@dataclass
class Cylinder:
    x_center: float
    y_center: float
    R: float
    w: float
    Type: str  # e.g., "cylinder"

    def __post_init__(self):
        if self.Type is None:
            self.Type = "cylinder"


@dataclass
class Atmosphere:
    density_0: float
    temperature_0: float
    pressure_0: float
    h: float
    R: float

    def __post_init__(self):
        self.density_0 = 1.225  # kg/m^3
        self.temperature_0 = 15  # ºC
        self.pressure_0 = 101325  #Pa
        self.R = 287.058  # J / (kg * k)

        self.pressure_h = self.pressure_0 * (1 - 0.0065 * (self.h / (self.temperature_0 + 273.15))) ** 5.2561
        self.temperature_h = self.temperature_0 - (6.5 * (self.h / 1000))

        self.temperature_h_in_K = self.temperature_h + 273.15

        self.density_h = self.pressure_h / (self.temperature_h_in_K * self.R)


@dataclass
class Airfoil:
    surfaceCoordinates: str
    centreCoordinates: Tuple[float, float]
    theta: float
    chord: float
    kuttaCondition: bool

    def __post_init__(self):
        import math as mt

        self.Type: str = "airfoil"
        self.theta_rad: float = mt.radians(self.theta)

        # Calcula el borde de ataque a partir del centro
        self.leading_edge, self.trailing_edge = self.centre_coordinates_to_leading_edge()
        # Carga el perfil original
        self.coords_original = self.Load_Airfoil(self.surfaceCoordinates)
        # Transforma (escala, rota y traslada) las coordenadas
        self.coords_transformed = self.scale_airfoil(
            self.coords_original,
            self.chord,
            self.leading_edge[0],
            self.leading_edge[1],
            self.theta_rad
        )
        self.save_scaled_coordinates(
            os.path.join(RESULTS_DIR, "Airfoil_txt", "NACA2412_ScaledCoordinatesTxt"))

    def centre_coordinates_to_leading_edge(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
            Calcula las coordenadas rotadas del borde de ataque y de salida
            usando el centro de cuerda y el ángulo de ataque.
            """
        x_center, y_center = self.centreCoordinates
        dx = (self.chord / 2) * np.cos(-self.theta_rad)
        dy = (self.chord / 2) * np.sin(-self.theta_rad)

        # Leading edge: desplazamiento hacia atrás
        x_leading = x_center - dx
        y_leading = y_center - dy

        # Trailing edge: desplazamiento hacia adelante
        x_trailing = x_center + dx
        y_trailing = y_center + dy

        return (x_leading, y_leading), (x_trailing, y_trailing)

    def Load_Airfoil(self, path: str) -> np.array:
        with open(path, 'r') as f:
            lines = f.readlines()
        coords = []
        for line in lines:
            try:
                x, y = map(float, line.strip().split())
                coords.append((x, y))
            except:
                continue
        return np.array(coords)

    def scale_airfoil(self, coords, chord, x0, y0, theta):
        """
           Escala, rota y traslada el perfil de acuerdo con el centro de cuerda (no borde de ataque).
           """
        x, y = coords[:, 0], coords[:, 1]
        x0, y0 = self.centreCoordinates

        # Escala
        x_scaled = x * chord
        y_scaled = y * chord

        # Encuentra el centro de la cuerda en el perfil original (usualmente (0.5, 0))
        # Este es el punto que queremos mover a (x_center, y_center)
        x_offset = 0.5 * chord
        y_offset = 0.0

        # Traslada el perfil para que el centro esté en (0, 0)
        x_translated = x_scaled - x_offset
        y_translated = y_scaled - y_offset

        # Rota alrededor del origen (centro de cuerda)
        x_rot = x_translated * np.cos(-theta) - y_translated * np.sin(-theta)
        y_rot = x_translated * np.sin(-theta) + y_translated * np.cos(-theta)

        # Traslada al centro deseado
        x_final = x_rot + x0
        y_final = y_rot + y0

        return np.column_stack((x_final, y_final))

    def save_scaled_coordinates(self, output_path: str):
        with open(output_path, 'w') as f:
            for x, y in self.coords_transformed:
                f.write(f"{x:.6f} {y:.6f}\n")

    def plot_airfoil(self, domain_size: float = 4.0):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.plot(self.coords_transformed[:, 0], self.coords_transformed[:, 1], 'b-', label='Scaled airfoil')
        plt.scatter(*self.centreCoordinates, color='red', label='Chord center')
        plt.plot(*self.trailing_edge, 'mo', label='Trailing edge')
        plt.xlim(0, domain_size)
        plt.ylim(0, domain_size)
        plt.gca().set_aspect('equal')
        plt.grid(True)
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title(f'Airfoil - Chord {self.chord} m, Angle {self.theta:.1f}°')
        plt.legend()
        plt.tight_layout()
        plt.show()
