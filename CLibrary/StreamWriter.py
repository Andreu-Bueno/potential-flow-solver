from .DataModels import *


def SaveComputations(filename: str, psi: np.ndarray, u: np.ndarray, v: np.ndarray, pressure: np.ndarray) -> int:
    path = os.path.join(RESULTS_DIR, "TXT")
    file: str = os.path.join(path, filename)
    try:
        with open(file, "w") as f:
            f.write("======== PSI ========\n")
            np.savetxt(f, psi, fmt="%18e")

            f.write("\n======== VELOCITY U========\n")
            np.savetxt(f, u, fmt="%18e")

            f.write("\n======== VELOCITY V ========\n")
            np.savetxt(f, v, fmt="%18e")

            f.write("\n======== PRESSURE ========\n")
            np.savetxt(f, pressure, fmt="%18e")

            print(f"Data saved in file with name: {filename}")
            return 0  # Éxito
    except Exception as e:
        print(f"Error saving computations: {e}")
        return 1  # Error


def SaveToNPZ(filename: str, psi: np.ndarray, u: np.ndarray, v: np.ndarray, pressure: np.ndarray) -> int:
    folder: str = os.path.join(RESULTS_DIR, "Numpy")
    path = os.path.join(folder, filename)

    try:
        np.savez(path, psi=psi, u=u, v=v, pressure=pressure)
        return 0
    except Exception as e:
        print(f'Error: {e}')
        return -1
