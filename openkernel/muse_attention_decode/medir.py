"""Mide submission.py en GPU como el juez: cada forma puntuada se captura en un CUDA graph y se repite con el L2 frio.
Da el tiempo por forma y el tiempo de cada carga (forma x peso), para afinar antes de gastar un `okc check`.

    python medir.py <carpeta de okc init> [--repeticiones 50] [--solo decode]

No compara con SGLang (su kernel no esta en Colab): compara contra el piso (mover los datos a 4.8 TB/s) y contra
tus propias corridas anteriores."""

import argparse
import importlib.util
import json
import os
import sys

import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
H200 = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}


def _modulo(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, archivo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _formas(carpeta):
    """Las 12 formas puntuadas: de formas.json si esta (lo escribe el cuaderno desde la API), si no, las de TASK.md."""
    archivo = os.path.join(carpeta, "formas.json")
    if os.path.exists(archivo):
        return json.load(open(archivo))
    formas = []
    for carga, batch in (("decode", 1), ("decode", 16), ("decode", 32), ("verify", 1), ("verify", 16), ("verify", 32)):
        for layer, weight, dist in (("sliding", 46, "soft"), ("full", 16, "coherent")):
            f = dict(batch=batch, ctx=80000, heads="tp8", layer=layer, page=1, dist=dist, weight=weight,
                     load=f"{carga} {batch}", seed=600)
            if carga == "verify":
                f["q_len"] = 8
            if batch > 1:
                f["spread"] = "agentic"
            formas.append(f)
    return formas


def medir(kernel_fn, data, repeticiones):
    basura = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")  # mas grande que el L2 (50 MB en H200)
    kernel_fn(data)  # compila
    torch.cuda.synchronize()
    grafo = torch.cuda.CUDAGraph()
    with torch.cuda.graph(grafo):
        kernel_fn(data)
    tiempos = []
    for _ in range(repeticiones):
        basura.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        grafo.replay()
        b.record()
        torch.cuda.synchronize()
        tiempos.append(a.elapsed_time(b) * 1000)
    tiempos.sort()
    return tiempos[len(tiempos) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("carpeta")
    p.add_argument("--repeticiones", type=int, default=50)
    p.add_argument("--solo", default="", help="solo las cargas que contengan este texto, p. ej. 'decode 32'")
    a = p.parse_args()
    sys.path.insert(0, a.carpeta)
    ref = _modulo("reference", os.path.join(a.carpeta, "reference.py"))
    sub = _modulo("submission", os.path.join(AQUI, "submission.py"))
    print(f"GPU: {torch.cuda.get_device_name()}")
    cargas = {}
    for f in _formas(a.carpeta):
        if a.solo not in f["load"]:
            continue
        caso = {k: v for k, v in f.items() if k not in ("weight", "load")}
        data = ref.generate_input(device="cuda", **caso)
        us = medir(sub.kernel_fn, data, a.repeticiones)
        piso = ref.speed_of_light_ns(hw=H200, **caso) / 1000
        cargas[f["load"]] = cargas.get(f["load"], 0.0) + us * f["weight"]
        print(f"{f['load']:10s} {f['layer']:8s}  {us:8.1f} us   piso H200 {piso:6.1f} us")
        del data
        torch.cuda.empty_cache()
    for carga, us in cargas.items():
        print(f"carga {carga:10s}  {us:9.1f} us por paso")


if __name__ == "__main__":
    main()
