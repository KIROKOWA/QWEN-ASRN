"""Compila sin GPU el CUDA de una version (CUDA_SRC, estilo vjeux / MuseOne17) para sm_90a: NVRTC (CUDA -> PTX) y
ptxas -v (registros, derrames, memoria compartida) de cada instancia de attn_kernel que lanza el anfitrion.

    pip install nvidia-cuda-nvcc-cu12==12.8.93 nvidia-cuda-nvrtc-cu12==12.8.93 nvidia-cuda-runtime-cu12==12.8.90 \
        nvidia-cuda-cccl-cu12
    python compilar_nvrtc.py [--archivo MuseOne17.py] [-D OKC_ALGO=1 ...]

Solo compila el codigo de dispositivo: quita los encabezados de torch y el anfitrion (launch / run), y agrega
instanciaciones explicitas de las plantillas. No importa el archivo (no hace falta torch): CUDA_SRC sale del AST."""
import ast
import ctypes
import glob
import os
import re
import subprocess
import sys
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))
NV = None
for raiz in sys.path:
    if os.path.isdir(os.path.join(raiz, "nvidia", "cuda_nvrtc")):
        NV = os.path.join(raiz, "nvidia")
        break

# las instancias que lanza launch<L>() (MODE: 1 split-KV, 2 clusters, 3 coop, 5/7 una pieza, 9/11 spec, 13/15 spec-una)
INSTANCIAS = [(L, "false", m) for L in (1, 8) for m in (1, 2, 3, 5, 7, 9, 11, 13, 15)] + [(L, "true", 0) for L in (1, 8)]


def fuente(archivo):
    arbol = ast.parse(open(os.path.join(AQUI, archivo)).read())
    for n in arbol.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "CUDA_SRC":
            return ast.literal_eval(n.value)
    raise SystemExit("sin CUDA_SRC")


def solo_dispositivo(src):
    src = re.sub(r"#include <torch/extension.h>\n|#include <ATen/cuda/CUDAContext.h>\n", "", src)
    # NVRTC no trae la biblioteca estandar de C++: libcu++ (cccl) da lo que usa el dispositivo
    src = src.replace("#include <type_traits>", "#include <cuda/std/type_traits>\nnamespace std { using cuda::std::integral_constant; }")
    src = src.replace("#include <stdint.h>", "#include <cuda/std/cstdint>\nusing cuda::std::uint8_t; using cuda::std::uint16_t; using cuda::std::uint32_t; using cuda::std::uint64_t; using cuda::std::int64_t; using cuda::std::int8_t;")
    src = src.replace("#include <cuda.h>", "#define INFINITY __int_as_float(0x7f800000)\n#define NAN __int_as_float(0x7fffffff)")
    corte = src.index("}  // namespace\n")
    cuerpo = src[:corte]
    inst = "\n".join(f"template __global__ void attn_kernel<{L}, {f}, {m}>(const __grid_constant__ Params);" for L, f, m in INSTANCIAS)
    return cuerpo + "\n" + inst + "\n}  // namespace\n"


def main():
    archivo = sys.argv[sys.argv.index("--archivo") + 1] if "--archivo" in sys.argv else "MuseOne17.py"
    defs = [a for i, a in enumerate(sys.argv) if i > 0 and sys.argv[i - 1] == "-D"]
    src = solo_dispositivo(fuente(archivo))
    # el namespace anonimo da nombres internos: NVRTC no los emite si nadie los usa; lo quitamos
    src = src.replace("namespace {", "namespace okc {", 1)
    nvrtc = ctypes.CDLL(glob.glob(os.path.join(NV, "cuda_nvrtc", "lib", "libnvrtc.so.12"))[0])
    prog = ctypes.c_void_p()
    r = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"k.cu", 0, None, None)
    assert r == 0, r
    incs = [os.path.join(NV, "cuda_runtime", "include"), os.path.join(NV, "cuda_nvcc", "include"),
            os.path.join(NV, "cuda_cccl", "include")]
    opts = [b"--gpu-architecture=compute_90a", b"-std=c++17", b"-default-device",
            b"-DNDEBUG", b"-lineinfo"]
    opts += [f"-I{i}".encode() for i in incs] + [f"-D{d}".encode() for d in defs]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    r = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
    n = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    log = ctypes.create_string_buffer(n.value)
    nvrtc.nvrtcGetProgramLog(prog, log)
    texto = log.value.decode(errors="replace")
    if texto.strip():
        print("\n".join(l for l in texto.splitlines() if "warning #" not in l and l.strip())[:6000])
    if r != 0:
        raise SystemExit(f"NVRTC fallo ({r})")
    nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(n))
    ptx = ctypes.create_string_buffer(n.value)
    nvrtc.nvrtcGetPTX(prog, ptx)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "k.ptx")
        open(p, "wb").write(ptx.value)
        ptxas = os.path.join(NV, "cuda_nvcc", "bin", "ptxas")
        out = subprocess.run([ptxas, "-arch=sm_90a", "-v", "-O3", p, "-o", os.path.join(d, "k.cubin")],
                             capture_output=True, text=True)
        if out.returncode:
            raise SystemExit(out.stderr[-6000:])
        nombre = None
        for l in out.stderr.splitlines():
            m = re.search(r"Compiling entry function '(\S+)'", l)
            if m:
                nombre = m.group(1)
                dm = re.search(r"attn_kernel(?:ILi|I)(\d)ELb(\d)ELi(\d+)E", nombre)
                nombre = f"attn_kernel<L={dm.group(1)}, flat={dm.group(2)}, mode={dm.group(3)}>" if dm else nombre
            if "registers" in l or "spill" in l:
                if "bytes stack frame" in l and " 0 bytes spill stores, 0 bytes spill loads" in l:
                    continue
                print(f"{nombre}: {l.strip()}")


if __name__ == "__main__":
    main()
