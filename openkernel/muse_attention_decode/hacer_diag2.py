"""Escribe <base>diag2.py: diagnostico por CTA para las capas completas y verify (MuseOne19, puntos 2 y 3).

Ademas de las fases de hacer_diag.py, cada CTA guarda su SM (%smid), cuantos segmentos y vueltas de 64 llaves hizo
y el tiempo total de sus bucles. El anfitrion imprime, por forma:
  - tiempo por vuelta (mediana / p90) = suma de bucles / vueltas: decode contra verify con los mismos bytes;
  - fin del bucle segun cuantos segmentos tuvo el CTA (los que cruzan una frontera de peticion) y segun si es el
    primero o el segundo CTA de su SM (por blockIdx);
  - los 6 CTAs que terminan mas tarde, con SM, segmentos, vueltas, entrada y duracion del bucle.
Las marcas solo se escriben con dbg (llamadas 1 y 2 fuera del grafo de cada forma con contexto >= 70K).

    python hacer_diag2.py [--base MuseOne17.py]

Ranuras por CTA (16 x int64): 0 entrada, 1 indices (1er segmento), 2 prologo (1er segmento), 3 fin del bucle
(ultimo), 4 merge, 5 barrera, 6 salida, 7 smid, 8 segmentos, 9 vueltas, 12 inicio del bucle en curso, 13 suma de
bucles (ns)."""
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))

MACRO = r'''#define PROF_DUMP
// ---- diagnostico por CTA (solo con p.dbg != nullptr: llamadas fuera del grafo) ----
#define DBG_I(k) ((blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + (k))
#define DBG_T(t) unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t))
#define STAMP(k) do { if (p.dbg && threadIdx.x == 0) { DBG_T(_t); p.dbg[DBG_I(k)] = (long long)_t; } } while (0)
#define STAMP1(k) do { if (p.dbg && threadIdx.x == 0 && p.dbg[DBG_I(k)] == 0) { DBG_T(_t); \
    p.dbg[DBG_I(k)] = (long long)_t; } } while (0)
#define LOOP_BEGIN() do { if (p.dbg && threadIdx.x == 0) { DBG_T(_t); p.dbg[DBG_I(12)] = (long long)_t; } } while (0)
#define LOOP_END(tiles) do { if (p.dbg && threadIdx.x == 0) { DBG_T(_t); p.dbg[DBG_I(3)] = (long long)_t; \
    p.dbg[DBG_I(13)] += (long long)_t - p.dbg[DBG_I(12)]; p.dbg[DBG_I(9)] += (tiles); p.dbg[DBG_I(8)] += 1; } } while (0)
#define STAMP_SM() do { if (p.dbg && threadIdx.x == 0) { unsigned _s; asm volatile("mov.u32 %0, %%smid;" : "=r"(_s)); \
    p.dbg[DBG_I(7)] = (long long)_s; } } while (0)'''

CAMBIOS = [
    ("#define PROF_DUMP", MACRO),
    ("  const int pair = blockIdx.z;\n  if (!FLAT) {", "  const int pair = blockIdx.z;\n  STAMP(0);\n  STAMP_SM();\n  if (!FLAT) {"),
    ("  cp_wait<IDXF ? 1 : 0>();\n  csync<WS, NT>();\n", "  cp_wait<IDXF ? 1 : 0>();\n  csync<WS, NT>();\n  STAMP1(1);\n"),
    ("  // ---- consumer state ----\n", "  STAMP1(2);\n  LOOP_BEGIN();\n  // ---- consumer state ----\n"),
    ("  // ---- merge: rescale to the fixed reference 2^0",
     "  LOOP_END((end - start + 63) >> 6);\n  // ---- merge: rescale to the fixed reference 2^0"),
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
        _diag_imprimir(cfg, plan, coop, grid, n_splits, dbg)
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


def _q(x, f):
    x = sorted(x)
    return x[min(len(x) - 1, int(f * len(x)))] if x else float("nan")


def _diag_imprimir(cfg, plan, coop, grid, n_splits, dbg):
    \"\"\"Fases, tiempo por vuelta y los CTAs lentos (sincroniza: solo fuera del grafo).\"\"\"
    t = dbg.view(grid, 16).cpu().tolist()
    vivos = [i for i in range(grid) if t[i][0] > 0]
    t0 = min(t[i][0] for i in vivos)
    us = lambda v: (v - t0) / 1000.0
    nombres = ("entrada", "indices", "prologo", "bucle", "merge", "barrera", "salida")
    partes = []
    for k, nom in enumerate(nombres):
        x = [us(t[i][k]) for i in vivos if t[i][k] > 0]
        if x:
            partes.append(f"{nom} {_q(x, .5):.2f}/{max(x):.2f}({len(x)})")
    print(f"diag2 B={cfg['batch']} L={cfg['q_len']} ventana={cfg['window']} plan={plan} coop={coop} grid={grid} | "
          + " ".join(partes), flush=True)
    con = [i for i in vivos if t[i][9] > 0]
    if not con:
        return
    pv = [t[i][13] / t[i][9] / 1000.0 for i in con]
    vueltas = [t[i][9] for i in con]
    print(f"  vuelta(us) mediana {_q(pv, .5):.3f} p10 {_q(pv, .1):.3f} p90 {_q(pv, .9):.3f} | vueltas por CTA "
          f"{min(vueltas)}..{max(vueltas)} | bucle total(us) mediana {_q([t[i][13] / 1000.0 for i in con], .5):.2f} "
          f"max {max(t[i][13] for i in con) / 1000.0:.2f}", flush=True)
    fin = {i: us(t[i][3]) for i in con}
    por_seg = {}
    for i in con:
        por_seg.setdefault(min(t[i][8], 3), []).append(fin[i])
    txt = " ".join(f"seg={k}{'+' if k == 3 else ''}: n={len(v)} med {_q(v, .5):.2f} max {max(v):.2f}"
                   for k, v in sorted(por_seg.items()))
    sm = {}
    for i in con:
        sm.setdefault(t[i][7], []).append(i)
    pri, seg = [], []
    for v in sm.values():
        v.sort()
        pri.append(fin[v[0]])
        seg.extend(fin[j] for j in v[1:])
    entradas = [us(t[i][0]) for i in con]
    print(f"  fin del bucle por segmentos: {txt} | 1er CTA del SM med {_q(pri, .5):.2f} max {max(pri):.2f}"
          + (f", 2o med {_q(seg, .5):.2f} max {max(seg):.2f}" if seg else "")
          + f" | SMs {len(sm)} | entrada p90 {_q(entradas, .9):.2f} max {max(entradas):.2f}", flush=True)
    lentos = sorted(con, key=lambda i: -fin[i])[:6]
    print("  lentos: " + "; ".join(
        f"cta {i} sm {t[i][7]} seg {t[i][8]} vueltas {t[i][9]} entra {us(t[i][0]):.2f} bucle {t[i][13] / 1000.0:.2f} "
        f"fin {fin[i]:.2f}" for i in lentos), flush=True)
"""


def main():
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "MuseOne17.py"
    texto = open(os.path.join(AQUI, base)).read()
    for viejo, nuevo in CAMBIOS + [(ANFITRION_VIEJO, ANFITRION_NUEVO), (RETORNO_VIEJO, RETORNO_NUEVO)]:
        n = texto.count(viejo)
        if n != 1:
            raise SystemExit(f"esperaba 1 aparicion y hay {n}: {viejo[:80]!r}")
        texto = texto.replace(viejo, nuevo)
    texto = texto.replace("# Cambios sobre la base: ninguno en el kernel CUDA (ver ANALISIS_MuseOne17.md).",
                          "# MuseOne19diag: diagnostico por CTA (hacer_diag2.py); el calculo no cambia.", 1)
    texto = texto.replace('name=f"okc_attn_v298{tag}_90a"', 'name=f"okc_attn_v298{tag}_90a_diag2"', 1)
    salida = os.path.join(AQUI, "MuseOne19diag.py")
    open(salida, "w").write(texto)
    print(f"escrito {salida}")


if __name__ == "__main__":
    main()
