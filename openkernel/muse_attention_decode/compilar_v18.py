"""Compila el kernel CUDA de v18 para sm_90a sin GPU: NVRTC (CUDA -> PTX) y ptxas -v (registros, derrames, memoria
compartida) para cada combinacion de plantilla que lanza el anfitrion. Atrapa errores antes de gastar un `okc check`.

    python compilar_v18.py [--archivo submission_v18.py] [--nvrtc ruta/libnvrtc.so.12] [--ptxas ruta/ptxas]

Necesita libnvrtc y ptxas de CUDA 12.8+ (pip: nvidia-cuda-nvrtc-cu12, nvidia-cuda-nvcc-cu12) y los encabezados de
nvidia-cuda-runtime-cu12."""
import ast
import ctypes
import glob
import os
import re
import subprocess
import sys
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))


def _arg(nombre, defecto):
    return sys.argv[sys.argv.index(nombre) + 1] if nombre in sys.argv else defecto


def _buscar(patron):
    for raiz in sys.path:
        hallado = glob.glob(os.path.join(raiz, patron))
        if hallado:
            return hallado[0]
    return None


def main():
    archivo = _arg("--archivo", "submission_v18.py")
    # sin importar el archivo (no hace falta torch ni triton): las constantes salen del AST
    arbol = ast.parse(open(os.path.join(AQUI, archivo)).read())
    valores = {n.targets[0].id: n.value for n in arbol.body if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name)}
    fuente_cuda = ast.literal_eval(valores["_V18_FUENTE"])
    config = ast.literal_eval(valores["_V18_CONFIG"])
    instancias = [f"{l}, {pag}, {str(rope).lower()}, {e}, {m}, {str(x16).lower()}"
                  for l, (e, m, x16) in config.items() for pag in (1, 64) for rope in (False, True)]
    nvrtc = ctypes.CDLL(_arg("--nvrtc", None) or _buscar("nvidia/cuda_nvrtc/lib/libnvrtc.so*"))
    ptxas = _arg("--ptxas", None) or _buscar("nvidia/cuda_nvcc/bin/ptxas")
    incluir = os.path.dirname(_buscar("nvidia/cuda_runtime/include/cuda_fp8.h"))
    fuente = fuente_cuda + "\n" + "\n".join(
        f"template __global__ void m18::attention<{i}>(const m18::Params);" for i in instancias)
    prog = ctypes.c_void_p()
    assert nvrtc.nvrtcCreateProgram(ctypes.byref(prog), fuente.encode(), b"k.cu", 0, None, None) == 0
    # las macros que pone load_inline de PyTorch
    opciones = [b"--gpu-architecture=compute_90a", b"-std=c++20", b"-default-device",
                b"-D__CUDA_NO_HALF_OPERATORS__", b"-D__CUDA_NO_HALF_CONVERSIONS__",
                b"-D__CUDA_NO_BFLOAT16_CONVERSIONS__", b"-D__CUDA_NO_HALF2_OPERATORS__", b"-I" + incluir.encode()]
    res = nvrtc.nvrtcCompileProgram(prog, len(opciones), (ctypes.c_char_p * len(opciones))(*opciones))
    n = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    log = ctypes.create_string_buffer(n.value)
    nvrtc.nvrtcGetProgramLog(prog, log)
    if log.value.strip():
        print(log.value.decode())
    if res != 0:
        sys.exit("NVRTC no compilo")
    nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(n))
    ptx = ctypes.create_string_buffer(n.value)
    nvrtc.nvrtcGetPTX(prog, ptx)
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
        f.write(ptx.value.decode())
    salida = subprocess.run([ptxas, "-v", "--gpu-name=sm_90a", f.name, "-o", os.devnull], capture_output=True,
                            text=True)
    if "--ptx" in sys.argv:
        print(open(f.name).read())
    os.unlink(f.name)
    texto = salida.stdout + salida.stderr
    for linea in texto.splitlines():
        if "error" in linea.lower() or "warning" in linea.lower():
            print(linea)
    for nombre, uso in re.findall(r"Compiling entry function '(\S+)'.*?\n(?:.*\n)*?.*?(Used \d+ registers[^\n]*)", texto):
        print(f"{nombre[:90]:90s} {uso}")
    for linea in texto.splitlines():
        if "spill" in linea and not re.search(r"0 bytes spill stores, 0 bytes spill loads", linea):
            print(linea)
    sys.exit(salida.returncode)


if __name__ == "__main__":
    main()
