"""Prueba submission.py contra el reference.py del reto, en casos chicos parecidos a las 16 pruebas del juez.

    python probar_local.py <carpeta de okc init>          # en GPU
    TRITON_INTERPRET=1 python probar_local.py <carpeta>    # en CPU, con el interprete de Triton (lento)

La carpeta es la que escribe `okc init muse_attention_decode <carpeta>` (reference.py y checks.py). Solo comprueba
que el resultado sea correcto; la velocidad la mide `okc check`."""

import importlib.util
import os
import sys
import time

import torch

AQUI = os.path.dirname(os.path.abspath(__file__))

CASOS = [
    dict(batch=1, ctx=10, heads="tp8", layer="sliding"),
    dict(batch=3, ctx=2047, heads="tp8", layer="sliding"),
    dict(batch=2, ctx=2500, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1),
    dict(batch=2, ctx=3000, heads="tp8", layer="full"),
    dict(batch=2, ctx=3000, heads="tp8", layer="sliding", dist="edges", page=1),
    dict(batch=2, ctx=3000, heads="tp8", layer="full", q_len=8, dist="edges"),
    dict(batch=2, ctx=3000, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=1 / 64),
    dict(batch=2, ctx=3000, heads="tp8", layer="full", kv_scale=1 / 64),
    dict(batch=1, ctx=4000, heads="dp", layer="full", q_len=8, dist="soft"),
    dict(batch=2, ctx=5000, heads="tp8", layer="full", dist="coherent", page=1),
    dict(batch=2, ctx=6000, heads="tp8", layer="sliding", q_len=8, dist="soft", page=1),
]


def _modulo(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, archivo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(carpeta):
    sys.path.insert(0, carpeta)  # reference.py importa checks
    ref = _modulo("reference", os.path.join(carpeta, "reference.py"))
    sub = _modulo("submission", os.path.join(AQUI, "submission.py"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "--grandes" in sys.argv:  # los tamaños de las pruebas del juez (solo en GPU)
        CASOS.extend([dict(batch=2, ctx=100000, heads="tp8", layer="full", q_len=8, page=1),
                      dict(batch=4, ctx=80000, heads="tp8", layer="full", spread="mixed", dist="coherent", page=1),
                      dict(batch=1, ctx=1048576, heads="tp8", layer="full", q_len=8)])
    fallas = 0
    for i, caso in enumerate(CASOS):
        data = ref.generate_input(seed=100 + i, device=device, **caso)
        t0 = time.time()
        out = sub.kernel_fn(data)
        ok, msg = ref.check_implementation(data, ref.collect(data, out))
        fallas += not ok
        print(f"{'ok   ' if ok else 'FALLA'} {caso}  {msg}  ({time.time() - t0:.1f} s)", flush=True)
    print("todo bien" if not fallas else f"{fallas} casos fallaron")
    return 1 if fallas else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "."))
