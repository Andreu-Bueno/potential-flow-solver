from .DataModels import *


def LoadData(filename: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(filename, "r") as f:
        lines = f.readlines()

    def extract_block(start_marker: str, lines: list[str], shape: tuple[int, int]) -> np.ndarray:
        start = next(i for i, line in enumerate(lines) if start_marker in line) + 1
        end = next((i for i in range(start, len(lines)) if lines[i].startswith("========")), len(lines))
        block_lines = lines[start:end]
        # Convertimos las líneas a un objeto tipo archivo para loadtxt
        from io import StringIO
        return np.loadtxt(StringIO("".join(block_lines))).reshape(shape)

    psi = extract_block("PSI", lines, shape)
    u = extract_block("VELOCITY U", lines, shape)
    v = extract_block("VELOCITY V", lines, shape)
    pressure = extract_block("PRESSURE", lines, shape)

    return psi, u, v, pressure


def LoadFromNPZ(filename: str) -> tuple[Any, Any, Any, Any] | tuple[None, None, None, None]:
    folder = RESULTS_DIR
    path = os.path.join(folder, filename)

    try:
        data = np.load(path)
        psi = data["psi"]
        u = data["u"]
        v = data["v"]
        pressure = data["pressure"]
        return psi, u, v, pressure
    except Exception as e:
        print(f"Error loading .npz file: {e}")
        return None, None, None, None
