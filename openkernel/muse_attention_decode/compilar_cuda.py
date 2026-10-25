"""Compila el kernel CUDA de una version (su _CUDA_FUENTE) para sm_90a sin GPU: NVRTC (CUDA -> PTX) y ptxas -v
(registros, derrames, memoria compartida). Atrapa los errores de compilacion antes de gastar un `okc check`.

    python compilar_cuda.py [--archivo submission_v11.py]

Necesita libnvrtc (viene con torch: nvidia/cu13/lib) y un ptxas de CUDA 12.9 (pip install nvidia-cuda-nvcc-cu12);
el ptxas de Triton no lee el PTX 9.0 de NVRTC 13, asi que la version declarada se baja a 8.8."""
import ctypes
import glob
import importlib.util
import os
import subprocess
import sys
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))


def main():
    archivo = sys.argv[sys.argv.index("--archivo") + 1] if "--archivo" in sys.argv else "submission_v12.py"
    spec = importlib.util.spec_from_file_location("sub", os.path.join(AQUI, archivo))
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    import nvidia as paquete
    raiz = list(paquete.__path__)[0]
    nvrtc = ctypes.CDLL(glob.glob(os.path.join(raiz, "cu13", "lib", "libnvrtc.so*"))[0])
    prog = ctypes.c_void_p()
    assert nvrtc.nvrtcCreateProgram(ctypes.byref(prog), sub._CUDA_FUENTE.encode(), b"k.cu", 0, None, None) == 0
    # Las mismas macros que pone load_inline de PyTorch (el juez compila asi): sin operadores ni conversiones implicitas
    # de half y bf16.
    opciones = [b"--gpu-architecture=compute_90a", b"-std=c++20", b"-default-device",
                b"-D__CUDA_NO_HALF_OPERATORS__", b"-D__CUDA_NO_HALF_CONVERSIONS__",
                b"-D__CUDA_NO_BFLOAT16_CONVERSIONS__", b"-D__CUDA_NO_HALF2_OPERATORS__",
                b"-I" + os.path.join(raiz, "cu13", "include").encode()]
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
    texto = ptx.value.decode().replace(".version 9.0", ".version 8.8", 1)
    ptxas = os.path.join(raiz, "cuda_nvcc", "bin", "ptxas")
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
        f.write(texto)
    salida = subprocess.run([ptxas, "-v", "--gpu-name=sm_90a", f.name, "-o", os.devnull], capture_output=True, text=True)
    os.unlink(f.name)
    print(salida.stdout + salida.stderr)
    sys.exit(salida.returncode)


if __name__ == "__main__":
    main()
