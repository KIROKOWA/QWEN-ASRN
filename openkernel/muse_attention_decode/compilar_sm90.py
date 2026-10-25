"""Compila _parcial y _final para sm_90 (H200) sin GPU, en las combinaciones de las pruebas del juez, y reporta por
kernel los registros, los bytes derramados a memoria local y la memoria compartida (ptxas -v), y cuantos programas
caben a la vez por SM. Atrapa los errores de compilacion (que el interprete de CPU no ve) antes de gastar un
`okc check`, y sirve para elegir configuraciones: con lecturas sueltas del cache, cuantos mas programas por SM, mas
lecturas en vuelo.

    python compilar_sm90.py [--archivo submission_v4.py]      # con torch y triton instalados; no hace falta GPU

Los punteros y los pasos se compilan con divisibilidad 16, como los especializa Triton al lanzar (sin eso las lecturas
FP8 no se vectorizan y los registros salen inflados)."""
import importlib.util
import itertools
import os
import re
import subprocess
import sys
import tempfile

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

AQUI = os.path.dirname(os.path.abspath(__file__))
OBJETIVO = GPUTarget("cuda", 90, 32)
PTXAS = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas")
TIPOS = dict(QKV="*bf16", SEQ="*i32", TABLA="*i32", LOC="*i64", KC="*fp8e4nv", VC="*fp8e4nv", KS="*fp32", VS="*fp32",
             MP="*fp32", LP="*fp32", AP="*fp32", OUT="*bf16", paso_qkv="i32", paso_tabla="i32", chunk="i32",
             n_pedazos="i32", n_trabajos="i32", n_lista="i32", min_chunk="i32", qk_escala="fp32", escala_log2="fp32",
             ACC="*fp32", SUMA="*fp32", CNT="*i32", c_ref="fp32")
DIVISIBLES = {"paso_qkv", "chunk", "min_chunk"}


def _modulo(archivo):
    spec = importlib.util.spec_from_file_location("submission", os.path.join(AQUI, archivo))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compilar(fn, constantes, opciones, rope):
    """Compila fn con las constantes que acepta (ignora las demas). Devuelve (registros, derramados, compartida)."""
    consts = {k: v for k, v in constantes.items() if k in fn.arg_names}
    tipos = dict(TIPOS, ROPE=rope)
    sig = {n: ("constexpr" if n in consts else tipos[n]) for n in fn.arg_names}
    div = [n for n in fn.arg_names if n not in consts and (sig[n].startswith("*") or n in DIVISIBLES)]
    src = ASTSource(fn=fn, signature=sig, constexprs={(fn.arg_names.index(k),): v for k, v in consts.items()},
                    attrs={(fn.arg_names.index(n),): [["tt.divisibility", 16]] for n in div})
    k = triton.compile(src, target=OBJETIVO, options=opciones)
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
        f.write(k.asm["ptx"])
    salida = subprocess.run([PTXAS, "-v", "--gpu-name=sm_90a", f.name, "-o", os.devnull],
                            capture_output=True, text=True).stderr
    os.unlink(f.name)
    regs = int(re.search(r"Used (\d+) registers", salida).group(1))
    derrame = re.search(r"(\d+) bytes spill stores", salida)
    return regs, int(derrame.group(1)) if derrame else 0, k.metadata.shared


def por_sm(regs, compartida, warps):
    return min(65536 // (max(regs, 24) * 32 * warps), (227 * 1024) // max(compartida, 1), 64 // warps, 32)


def main_v6(sub):
    """v6: un solo kernel (_atencion) con la configuracion por forma de sub.CONFIG."""
    for (hq, hkv), largo, pagina, deslizante in itertools.product([(16, 2), (128, 16)], [1, 8], [1, 64], [True, False]):
        g = hq // hkv
        bn, etapas, warps, _, grupos, prefetch, qk_fp8, _, _ = sub.CONFIG[(largo, not deslizante)]
        filas = max(16, triton.next_power_of_2(largo * g // grupos))
        constantes = dict(B=4, BB=4, HQ=hq, HKV=hkv, L=largo, G=g, R=grupos, FILAS=filas, PAGINA=pagina,
                          VENTANA=2048 if deslizante else 0, CON_ROPE=deslizante, BLOQUE_N=bn, ETAPAS=etapas,
                          BLOQUE_L=max(16, triton.next_power_of_2(largo)), PREFETCH=prefetch, QK_FP8=qk_fp8,
                          QUORUM=True, EN_CPU=False)
        rope = "*fp32" if deslizante else "*bf16"
        variantes = [False, True] if "RANURAS" in sub._atencion.arg_names else [None]
        for ranuras in variantes:
            c = dict(constantes, RANURAS=bool(ranuras))
            r, dr, sh = compilar(sub._atencion, c, dict(num_warps=warps), rope)
            r2, dr2, _ = compilar(sub._cierre, c, dict(num_warps=4), rope)
            print(f"hq={hq:3d} L={largo} pagina={pagina:2d} {'ventana ' if deslizante else 'completa'}"
                  f"{' ranuras' if ranuras else ''}: _atencion {r:3d} regs {dr:4d} B derramados {sh:6d} B compartida "
                  f"-> {por_sm(r, sh, warps)}/SM | _cierre {r2} regs {dr2} B derr.", flush=True)
        if hasattr(sub, "_parcial_v2"):  # el metodo de pedazos (v8/v9)
            bm = max(16, triton.next_power_of_2(largo * g))
            bn2 = 64 if deslizante else 128
            for etapas in (2, 3):
                c = dict(HQ=hq, HKV=hkv, L=largo, G=g, PAGINA=pagina, VENTANA=2048 if deslizante else 0,
                         CON_ROPE=deslizante, BLOQUE_M=bm, BLOQUE_N=bn2, ETAPAS=etapas, EN_CPU=False)
                r, dr, sh = compilar(sub._parcial_v2, c, dict(num_warps=4, num_stages=2), rope)
                print(f"    _parcial_v2 etapas={etapas}: {r} regs {dr} B derramados {sh} B -> {por_sm(r, sh, 4)}/SM")
            r, dr, _ = compilar(sub._final_v2, dict(c, BLOQUE_S=64, BLOQUE_L=triton.next_power_of_2(largo)),
                                dict(num_warps=4), rope)
            print(f"    _final_v2: {r} regs {dr} B derramados", flush=True)
    print("todas las combinaciones compilan para sm_90")


def main():
    archivo = sys.argv[sys.argv.index("--archivo") + 1] if "--archivo" in sys.argv else "submission.py"
    sub = _modulo(archivo)
    print(f"{archivo}: sm_90")
    if hasattr(sub, "_atencion"):
        return main_v6(sub)
    for (hq, hkv), largo, pagina, deslizante in itertools.product([(16, 2), (128, 16)], [1, 8], [1, 64], [True, False]):
        g = hq // hkv
        bm = max(16, triton.next_power_of_2(largo * g))
        # Lo que kernel_fn elegiria para esta forma.
        etapas = 1
        if hasattr(sub, "CONFIG") and (largo, not deslizante) in sub.CONFIG:
            bn, par, warps, etapas, _ = sub.CONFIG[(largo, not deslizante)]
        elif hasattr(sub, "CONFIG"):
            bn, par, warps, _ = sub.CONFIG.get(largo, sub.CONFIG[8])
        else:
            bn, par, warps = (64 if deslizante else 128), (2 if hkv % 2 == 0 else 1), (8 if bm > 16 else 4)
        if hkv % par:
            par = 1
        constantes = dict(B=4, BB=4, HQ=hq, HKV=hkv, L=largo, G=g, PAGINA=pagina,
                          VENTANA=2048 if deslizante else 0, CON_ROPE=deslizante, BLOQUE_M=bm, BLOQUE_N=bn, PAR=par,
                          ETAPAS=etapas, EN_CPU=False, BLOQUE_L=triton.next_power_of_2(largo))
        rope = "*fp32" if deslizante else "*bf16"
        r, dr, sh = compilar(sub._parcial, constantes, dict(num_warps=warps, num_stages=1), rope)
        lineas = [f"_parcial {r:3d} regs {dr:4d} B derramados {sh:6d} B compartida -> {por_sm(r, sh, warps)}/SM"]
        for bs in (32, 128):
            w = 8 if bs > 32 else 4
            r2, dr2, sh2 = compilar(sub._final, dict(constantes, BLOQUE_S=bs), dict(num_warps=w), rope)
            lineas.append(f"_final(S={bs}) {r2:3d} regs {dr2:3d} B derr.")
        print(f"hq={hq:3d} L={largo} pagina={pagina:2d} {'ventana ' if deslizante else 'completa'}: " + " | ".join(lineas),
              flush=True)
    print("todas las combinaciones compilan para sm_90")


if __name__ == "__main__":
    main()
