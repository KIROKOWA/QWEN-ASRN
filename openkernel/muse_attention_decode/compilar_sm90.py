"""Compila _parcial y _final para sm_90 (H200) sin GPU, en las combinaciones de las pruebas del juez: atrapa los
errores de compilacion (que el interprete de CPU no ve) antes de gastar un `okc check`.

    python compilar_sm90.py          # con torch y triton instalados; no hace falta GPU"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triton  # noqa: E402
import submission as sub  # noqa: E402
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
target = GPUTarget("cuda", 90, 32)

def compilar(fn, tipos, consts, opciones):
    sig = {}
    for n in fn.arg_names:
        sig[n] = 'constexpr' if n in consts else tipos[n]
    src = ASTSource(fn=fn, signature=sig, constexprs={(fn.arg_names.index(k),): v for k, v in consts.items()})
    return triton.compile(src, target=target, options=opciones)

n = 0
for (hq, hkv), L, pagina, deslizante in itertools.product([(16, 2), (128, 16)], [1, 8], [1, 64], [True, False]):
    g = hq // hkv
    bm = max(16, triton.next_power_of_2(L * g))
    rope_t = '*fp32' if deslizante else '*bf16'
    comunes = dict(QKV='*bf16', SEQ='*i32', KC='*fp8e4nv', VC='*fp8e4nv', KS='*fp32', ROPE=rope_t,
                   MP='*fp32', LP='*fp32', AP='*fp32', paso_qkv='i32', n_pedazos='i32', qk_escala='fp32',
                   escala_log2='fp32')
    k1 = compilar(sub._parcial, dict(comunes, TABLA='*i32', paso_tabla='i32', chunk='i32'),
                  dict(HQ=hq, HKV=hkv, L=L, G=g, PAGINA=pagina, VENTANA=2048 if deslizante else 0,
                       CON_ROPE=deslizante, BLOQUE_M=bm, BLOQUE_N=64 if deslizante else 128, PAR=2, EN_CPU=False),
                  dict(num_warps=8 if bm > 16 else 4, num_stages=2))
    for bs in (1, 32, 128):
        k2 = compilar(sub._final, dict(comunes, LOC='*i64', VS='*fp32', OUT='*bf16'),
                      dict(HQ=hq, HKV=hkv, L=L, G=g, CON_ROPE=deslizante, BLOQUE_M=bm, BLOQUE_S=bs, BLOQUE_L=L),
                      dict(num_warps=8 if bs > 32 else 4))
    n += 1
    print(f"ok hq={hq} L={L} pagina={pagina} {'deslizante' if deslizante else 'completa'}: "
          f"_parcial {k1.metadata.shared} B shared, {k1.n_regs if hasattr(k1,'n_regs') else '?'} regs", flush=True)
print(n, "combinaciones compiladas para sm_90")
