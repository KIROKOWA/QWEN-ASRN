"""Barrido de planes de lanzamiento de MuseOne17 en las 12 formas puntuadas, en una GPU Hopper (H100 / H200).

El kernel CUDA no cambia: solo se prueban otros planes (n_splits, nc, coop, cs) a traves de los parametros que
`_launch(data, **override)` ya acepta. Cada plan se compara contra el reference.py del reto antes de medirlo, y se
mide como el juez: CUDA graph, L2 frio (se escribe un bufer de 256 MB antes de cada repeticion), mediana. Los planes
de una forma se miden intercalados (ronda por ronda), asi la deriva de reloj de la GPU afecta a todos por igual.

    python barrido_plan.py <carpeta con reference.py y checks.py> [--archivo MuseOne17.py] [--rondas 40]
                           [--solo "decode 1"] [--capa sliding|full]

Imprime, por forma, cada plan con su mediana (us) y la razon contra el plan por defecto, y al final un _TUNED
sugerido con los planes que ganan por mas del 2 %. En una H100 (3.35 TB/s contra 4.8 de la H200) las formas limitadas
por ancho de banda (completas con 16/32 peticiones) no se trasladan tal cual; las de latencia (ventana, B = 1) si."""

import argparse
import importlib.util
import json
import os
import sys

import torch

AQUI = os.path.dirname(os.path.abspath(__file__))

FORMAS = []
for _carga, _b in (("decode", 1), ("decode", 16), ("decode", 32), ("verify", 1), ("verify", 16), ("verify", 32)):
    for _capa, _peso, _dist in (("sliding", 46, "soft"), ("full", 16, "coherent")):
        _f = dict(batch=_b, ctx=80000, heads="tp8", layer=_capa, page=1, dist=_dist, weight=_peso,
                  load=f"{_carga} {_b}", seed=600 + len(FORMAS) + 1)
        if _carga == "verify":
            _f["q_len"] = 8
        if _b > 1:
            _f["spread"] = "agentic"
        FORMAS.append(_f)


def _modulo(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, archivo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def candidatos(f, sms):
    """Planes a probar para una forma. {} es el plan por defecto (con el _TUNED del archivo)."""
    B, L, ventana = f["batch"], f.get("q_len", 1), f["layer"] == "sliding"
    planes = [{}]
    if ventana:
        tiles = (2048 + L - 1 + 63) // 64          # 32 (decode) / 33 (verify)
        por_req = sorted({n for n in (2, 3, 4, 5, 6, 8, 11, 16, 22, 32, 33) if n <= tiles and n * B <= 3 * sms})
        for n in por_req:
            for coop in (0, 1):
                planes.append(dict(n_splits=n, nc=1, coop=coop))
        if B == 1 and L == 1:
            for n in (16, 32):
                planes.append(dict(n_splits=n, nc=2, coop=0))
    elif B == 1:
        for n in (132, 198, 252, 264, 330, 396):
            for nc in (4, 8):
                for coop in (0, 1):
                    planes.append(dict(n_splits=n, nc=nc, coop=coop))
    else:
        # plano (flat): n_splits CTAs por par de cabezas kv sobre todas las peticiones
        for n in (sms, 2 * sms - 8, 2 * sms, 2 * sms + 8, 3 * sms, 4 * sms):
            for coop in (0, 1):
                planes.append(dict(n_splits=n, coop=coop))
    vistos, unicos = set(), []
    for p in planes:
        k = json.dumps(p, sort_keys=True)
        if k not in vistos:
            vistos.add(k)
            unicos.append(p)
    return unicos


def grafo(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("carpeta")
    p.add_argument("--archivo", default="MuseOne17.py")
    p.add_argument("--rondas", type=int, default=40)
    p.add_argument("--solo", default="")
    p.add_argument("--capa", default="")
    a = p.parse_args()
    sys.path.insert(0, a.carpeta)
    ref = _modulo("reference", os.path.join(a.carpeta, "reference.py"))
    sub = _modulo("submission", os.path.join(AQUI, a.archivo))
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"GPU: {torch.cuda.get_device_name()} ({sms} SMs)  archivo: {a.archivo}", flush=True)
    basura = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    vacio = grafo(lambda: None)
    sugerido = {}
    for f in FORMAS:
        if a.solo not in f["load"] or a.capa not in f["layer"]:
            continue
        caso = {k: v for k, v in f.items() if k not in ("weight", "load")}
        data = ref.generate_input(device="cuda", **caso)
        planes = candidatos(f, sms)
        grafos, nombres = [], []
        for plan in planes:
            fn = (lambda pl=plan: sub._launch(data, **pl)) if plan else (lambda: sub.kernel_fn(data))
            try:
                out = fn()
                ok, msg = ref.check_implementation(data, ref.collect(data, out))
            except Exception as e:   # un plan que el kernel no admite
                ok, msg = False, f"{type(e).__name__}: {e}"
            if not ok:
                print(f"  {f['load']} {f['layer']} plan {plan}: INCORRECTO ({str(msg)[:200]})", flush=True)
                continue
            grafos.append(grafo(fn))
            nombres.append(plan)
        tiempos = [[] for _ in grafos]
        for _ in range(a.rondas):
            for i, g in enumerate([vacio] + grafos):
                basura.zero_()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                g.replay()
                e1.record()
                torch.cuda.synchronize()
                if i == 0:
                    t_vacio = e0.elapsed_time(e1) * 1000
                else:
                    tiempos[i - 1].append(e0.elapsed_time(e1) * 1000 - t_vacio)
        med = [sorted(t)[len(t) // 2] for t in tiempos]
        base = med[0]
        print(f"\n{f['load']} {f['layer']} (peso {f['weight']}): por defecto {base:.2f} us", flush=True)
        orden = sorted(range(len(med)), key=lambda i: med[i])
        for i in orden[:12]:
            print(f"  {med[i]:8.2f} us  x{base / med[i]:.3f}  {nombres[i] or 'por defecto'}")
        mejor = orden[0]
        if nombres[mejor] and base / med[mejor] > 1.02:
            clave = (f["batch"], f.get("q_len", 1), f["layer"] == "sliding", 1)
            sugerido[clave] = nombres[mejor]
        del data, grafos
        torch.cuda.empty_cache()
    print("\n_TUNED sugerido (solo lo que gana > 2 %; confirmalo con okc check):")
    for k, v in sugerido.items():
        print(f"    {k}: dict({', '.join(f'{a}={b}' for a, b in v.items())}),")


if __name__ == "__main__":
    main()
