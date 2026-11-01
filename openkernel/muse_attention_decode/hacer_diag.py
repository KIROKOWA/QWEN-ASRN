"""Escribe <base>diag.py: la misma version con marcas de tiempo por fase y por CTA (%globaltimer), para ver en la H200
del juez donde se va el tiempo de cada forma. Las marcas solo se escriben cuando el anfitrion pasa el tensor dbg, y el
anfitrion solo lo pasa en llamadas fuera del CUDA graph (pruebas y calentamiento), dos veces por forma grande; las
repeticiones medidas corren el mismo codigo con dbg = nullptr (una comparacion por marca, sin escrituras).

    python hacer_diag.py [--base MuseOne17.py]

Fases (us desde el primer CTA en entrar): 0 entrada, 1 indices de pagina listos, 2 prologo listo (q, piezas nuevas),
3 fin del bucle, 4 merge emitido, 5 barrera / ultimo CTA, 6 salida escrita."""
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))

MACRO = r'''#define PROF_DUMP
// ---- marcas de tiempo por fase (solo con p.dbg != nullptr: llamadas de diagnostico fuera del grafo) ----
#define STAMP(k) do { if (p.dbg && threadIdx.x == 0) { unsigned long long _t; \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t)); \
    p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + (k)] = (long long)_t; } } while (0)'''

CAMBIOS = [
    ("#define PROF_DUMP", MACRO),
    ("  const int pair = blockIdx.z;\n  if (!FLAT) {", "  const int pair = blockIdx.z;\n  STAMP(0);\n  if (!FLAT) {"),
    ("  cp_wait<IDXF ? 1 : 0>();\n  csync<WS, NT>();\n", "  cp_wait<IDXF ? 1 : 0>();\n  csync<WS, NT>();\n  STAMP(1);\n"),
    ("  // ---- consumer state ----\n", "  STAMP(2);\n  // ---- consumer state ----\n"),
    ("  // ---- merge: rescale to the fixed reference 2^0", "  STAMP(3);\n  // ---- merge: rescale to the fixed reference 2^0"),
    ("  // (last-CTA finalizes): no per-thread fences;", "  STAMP(4);\n  // (last-CTA finalizes): no per-thread fences;"),
    ("    csync<WS, NT>();\n    const int r_lo = split_idx * R,", "    csync<WS, NT>();\n    STAMP(5);\n    const int r_lo = split_idx * R,"),
    ("    }\n    \n  } else {\n  if (tid == 0) {\n    const int mine = end - start;",
     "    }\n    STAMP(6);\n  } else {\n  if (tid == 0) {\n    const int mine = end - start;"),
    ("  csync<WS, NT>();\n  if (S.flag) {\n", "  csync<WS, NT>();\n  if (S.flag) {\n  STAMP(5);\n"),
    ("  if (p.dbg && tid == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + 6] = 1;",
     "  STAMP(6);"),
    ("  csync<WS, NT>();\n  const int pairs = p.Hkv >> 1;\n  const int rows = p.B * pairs * L * 16;",
     "  csync<WS, NT>();\n  STAMP(5);\n  const int pairs = p.Hkv >> 1;\n  const int rows = p.B * pairs * L * 16;"),
    ("    if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;\n  }\n  \n}",
     "    if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;\n  }\n  STAMP(6);\n}"),
]

ANFITRION_VIEJO = """    (_mod or _module()).run(qkv, data["seq_lens"],"""
ANFITRION_NUEVO = """    diag = dbg is None and _diag_toca(cfg, page_table, P)
    if diag:
        grid = n_splits * (1 if flat else B) * pairs
        dbg = torch.zeros(grid * 16, dtype=torch.int64, device=device)
    (_mod or _module()).run(qkv, data["seq_lens"],"""
RETORNO_VIEJO = """                  dbg, cs, ncl, tma)
    return out
"""
RETORNO_NUEVO = """                  dbg, cs, ncl, tma)
    if diag:
        _diag_imprimir(cfg, plan, coop, grid, dbg)
    return out


_DIAG = {}
DIAG_LLAMADAS = (1, 2)   # que llamadas fuera del grafo se perfilan por forma (la 0 compila y calienta)


def _diag_toca(cfg, page_table, P):
    \"\"\"Solo formas grandes (contexto >= 70K: las puntuadas), fuera de la captura, las llamadas DIAG_LLAMADAS.\"\"\"
    if page_table.shape[1] * P < 70000 or torch.cuda.is_current_stream_capturing():
        return False
    clave = (cfg["batch"], cfg["q_len"], cfg["window"])
    n = _DIAG.get(clave, 0)
    _DIAG[clave] = n + 1
    return n in DIAG_LLAMADAS


def _diag_imprimir(cfg, plan, coop, grid, dbg):
    \"\"\"Por fase: mediana / maximo en us desde el primer CTA en entrar (sincroniza: solo fuera del grafo).\"\"\"
    t = dbg.view(grid, 16)[:, :7].cpu()
    vivos = t[:, 0] > 0
    t0 = int(t[vivos, 0].min())
    nombres = ("entrada", "indices", "prologo", "bucle", "merge", "barrera", "salida")
    partes = []
    for i, nom in enumerate(nombres):
        m = vivos & (t[:, i] > 0)
        if not bool(m.any()):
            continue
        x = (t[m, i] - t0).double() / 1000
        partes.append(f"{nom} {x.median().item():.2f}/{x.max().item():.2f}({int(m.sum())})")
    fin = int(t[vivos].max())
    print(f"diag B={cfg['batch']} L={cfg['q_len']} ventana={cfg['window']} plan={plan} coop={coop} grid={grid} | "
          + " ".join(partes) + f" | span {(fin - t0) / 1000:.2f} us", flush=True)
"""


def main():
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "MuseOne17.py"
    texto = open(os.path.join(AQUI, base)).read()
    for viejo, nuevo in CAMBIOS + [(ANFITRION_VIEJO, ANFITRION_NUEVO), (RETORNO_VIEJO, RETORNO_NUEVO)]:
        n = texto.count(viejo)
        if n != 1:
            raise SystemExit(f"esperaba 1 aparicion y hay {n}: {viejo[:80]!r}")
        texto = texto.replace(viejo, nuevo)
    texto = texto.replace("# Cambios sobre la base: ninguno en el kernel CUDA",
                          "# DIAGNOSTICO: marcas de tiempo por fase (hacer_diag.py); el calculo no cambia", 1)
    texto = texto.replace('name=f"okc_attn_v298{tag}_90a"', 'name=f"okc_attn_v298{tag}_90a_diag"', 1)
    salida = os.path.join(AQUI, base.replace(".py", "diag.py"))
    open(salida, "w").write(texto)
    print(f"escrito {salida}")


if __name__ == "__main__":
    main()
