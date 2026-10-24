"""Entrenar las capas ASRN injertadas en Qwen 2.5 (Qwen congelado en bf16, ASRN en fp32 con AdamW).

    python -m qwen_asrn.entrenar --datos piezas.u32 --validacion val.u32 --guardar_en qwen_asrn.pt
    python -m qwen_asrn.entrenar ... --reanudar qwen_asrn.pt          # sigue exacto donde quedo

Datos: los numeros de pieza del tokenizador de Qwen ya tokenizados, en un archivo que lee asrn.leer_piezas (.u32/.i32
int32 crudo o .pt con un tensor 1D; el vocabulario de Qwen, 152 064, no cabe en uint16).

Por paso: `acumular` lotes de `lote` ventanas de `largo` piezas. AdamW (asrn.crear_optimizador: decaimiento solo en
matrices) sobre los parametros de ASRN y nada mas; recorte dinamico del gradiente por grupos (asrn.RecorteDinamico)
o recorte fijo; calentamiento + coseno (asrn.tasa_aprendizaje); un paso con perdida o norma no finita se salta. Antes
del primer paso comprueba el arranque neutro (el injerto no cambia nada de Qwen, bit a bit) y se detiene si falla.
Guarda solo los pesos de ASRN (fp32) mas el estado de AdamW, el paso y el azar (para reanudar); el mejor en validacion
(<archivo>.mejor.pt), solo los pesos.
"""
import argparse
import math
import os
import time

import torch

import asrn

from .modelo import QwenASRN, cargar_asrn, cargar_qwen, guardar_asrn, verificar_arranque_neutro


def _lote(datos, largo, lote, g, disp):
    ini = torch.randint(0, datos.numel() - largo, (lote,), generator=g)
    idx = (ini.unsqueeze(1) + torch.arange(largo + 1)).to(datos.device)
    w = datos[idx].to(disp, non_blocking=True).long()
    return w[:, :-1], w[:, 1:]


def _preparar(datos, V, largo, disp):
    datos = torch.as_tensor(datos).reshape(-1)
    if datos.is_floating_point() or datos.numel() <= largo + 1:
        raise ValueError(f"hacen falta mas de {largo + 1} numeros de pieza enteros (hay {datos.numel()})")
    lo, hi = int(datos.min()), int(datos.max())
    if lo < 0 or hi >= V:
        raise ValueError(f"piezas fuera de [0, {V}): minimo {lo}, maximo {hi} (¿tokenizador de otro modelo?)")
    datos = datos.to(torch.int32)
    return datos.to(disp) if disp.type == 'cuda' and datos.numel() * 4 < 2 ** 31 else datos


@torch.no_grad()
def evaluar(modelo, datos, largo, lote, max_lotes, sin_asrn=False):
    """Perdida media sobre ventanas seguidas desde el inicio de `datos` (sin_asrn=True: la de Qwen solo)."""
    estaba = modelo.training
    modelo.eval()
    disp = modelo.qwen.model.embed_tokens.weight.device
    total, n = 0.0, 0
    try:
        for b in range(max_lotes):
            ini = torch.arange(lote) * largo + b * lote * largo
            if int(ini[-1]) + largo + 1 > datos.numel():
                break
            idx = (ini.unsqueeze(1) + torch.arange(largo + 1)).to(datos.device)
            w = datos[idx].to(disp).long()
            total += float(modelo.perdida(w[:, :-1], w[:, 1:], sin_asrn=sin_asrn)) * w[:, 1:].numel()
            n += w[:, 1:].numel()
    finally:
        modelo.train(estaba)
    return total / max(n, 1)


def entrenar(modelo, datos, validacion=None, largo=4096, lote=1, acumular=16, pasos=2000, lr=2e-4, lr_min=None,
             calentamiento=100, decaimiento=0.1, recorte='dinamico', evaluar_cada=250, lotes_eval=20, guardar_en=None,
             guardar_cada=250, reanudar=None, semilla=0, registrar_cada=10, verificar_neutro=True, max_saltos=20,
             registro=print):
    disp = modelo.qwen.model.embed_tokens.weight.device
    lr_min = lr / 10 if lr_min is None else lr_min
    datos = _preparar(datos, modelo.V, largo, disp)
    if validacion is not None:
        validacion = _preparar(validacion, modelo.V, largo, disp)
    if disp.type == 'cuda':            # como asrn.entrenar: el reloj no usa TF32 de todos modos (_ProyeccionIEEE)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    modelo.revisar_contrato(qwen_tambien=True)
    opt = asrn.crear_optimizador(modelo, lr=lr, decaimiento=decaimiento)
    recortar = asrn.RecorteDinamico(modelo) if recorte == 'dinamico' else None
    params = modelo.parametros_entrenables()
    g = torch.Generator().manual_seed(semilla)
    paso, mejor = 0, math.inf
    if reanudar:
        r = cargar_asrn(modelo, reanudar)
        if 'opt' in r:
            opt.load_state_dict(r['opt'])
            paso, mejor = r['paso'], r.get('mejor', math.inf)
            g.set_state(r['azar'])
        registro(f'reanudado desde {reanudar}, paso {paso}')
    elif verificar_neutro:
        x, _ = _lote(datos, min(largo, 2 * modelo.ventana), 1, torch.Generator().manual_seed(semilla + 1), disp)
        ok, det = verificar_arranque_neutro(modelo, x)
        registro(f'arranque neutro: {det}')
        if not ok:
            raise RuntimeError(f'el injerto cambia las activaciones de Qwen en el paso 0: {det}')

    def guardar(archivo, para_reanudar=True):
        """Con para_reanudar, tambien AdamW (2x los pesos), el paso y el azar; si no, solo los pesos de ASRN."""
        extra = dict(opt=opt.state_dict(), paso=paso, mejor=mejor, azar=g.get_state()) if para_reanudar else {}
        guardar_asrn(modelo, archivo, **extra)

    modelo.train()
    saltos, t0, piezas = 0, time.time(), 0
    if disp.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    try:
        while paso < pasos:
            tasa = asrn.tasa_aprendizaje(paso, lr, pasos, calentamiento, lr_min)
            for grupo in opt.param_groups:
                grupo['lr'] = tasa
            opt.zero_grad(set_to_none=True)
            perdida = torch.zeros((), device=disp)
            for _ in range(acumular):
                x, y = _lote(datos, largo, lote, g, disp)
                p = modelo.perdida(x, y)
                (p / acumular).backward()
                perdida += p.detach() / acumular
            norma = recortar() if recortar else torch.nn.utils.clip_grad_norm_(params, recorte or float('inf'))
            if bool(torch.isfinite(perdida) & torch.isfinite(norma)):
                opt.step()
                saltos = 0
            else:
                saltos += 1
                registro(f'paso {paso}: perdida o norma no finita, se salta ({saltos} seguidos)')
                if saltos > max_saltos:
                    raise RuntimeError(f'{saltos} pasos seguidos con NaN/inf')
            paso += 1
            piezas += acumular * lote * largo
            if paso % registrar_cada == 0 or paso == 1:
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if disp.type == 'cuda' else 0.0
                registro(f'paso {paso} perdida {float(perdida):.4f} norma {float(norma):.3f} lr {tasa:.2e} '
                         f'{piezas / dt:,.0f} piezas/s memoria max {mem:.1f} GiB')
                t0, piezas = time.time(), 0
            if validacion is not None and paso % evaluar_cada == 0:
                pv = evaluar(modelo, validacion, largo, lote, lotes_eval)
                registro(f'paso {paso} validacion {pv:.4f} (ppl {math.exp(min(pv, 50)):.2f})')
                if pv < mejor and guardar_en:
                    mejor = pv
                    guardar(os.path.splitext(guardar_en)[0] + '.mejor.pt', para_reanudar=False)
            if guardar_en and paso % guardar_cada == 0:
                guardar(guardar_en)
    except KeyboardInterrupt:
        registro('interrumpido: guardando')
    if guardar_en:
        guardar(guardar_en)
    return modelo


def principal(args=None):
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--modelo', default='Qwen/Qwen2.5-7B')
    a.add_argument('--datos', required=True)
    a.add_argument('--validacion')
    a.add_argument('--largo', type=int, default=4096)
    a.add_argument('--lote', type=int, default=1)
    a.add_argument('--acumular', type=int, default=16)
    a.add_argument('--pasos', type=int, default=2000)
    a.add_argument('--lr', type=float, default=2e-4)
    a.add_argument('--lr_min', type=float)
    a.add_argument('--calentamiento', type=int, default=100)
    a.add_argument('--decaimiento', type=float, default=0.1)
    a.add_argument('--recorte', default='dinamico', help="'dinamico' (asrn.RecorteDinamico) o una norma maxima")
    a.add_argument('--ventana', type=int, default=1024)
    a.add_argument('--sumideros', type=int, default=4)
    a.add_argument('--largo_cada', type=int, default=2)
    a.add_argument('--checkpoint', default='asrn', choices=('no', 'asrn', 'todo'))
    a.add_argument('--trozo_perdida', type=int, default=1024)
    a.add_argument('--evaluar_cada', type=int, default=250)
    a.add_argument('--lotes_eval', type=int, default=20)
    a.add_argument('--guardar_en', default='qwen_asrn.pt')
    a.add_argument('--guardar_cada', type=int, default=250)
    a.add_argument('--reanudar')
    a.add_argument('--semilla', type=int, default=0)
    a.add_argument('--registrar_cada', type=int, default=10)
    a.add_argument('--dispositivo', default='cuda')
    a = a.parse_args(args)
    torch.manual_seed(a.semilla)
    qwen = cargar_qwen(a.modelo, a.dispositivo)
    modelo = QwenASRN(qwen, ventana=a.ventana, sumideros=a.sumideros, largo_cada=a.largo_cada,
                      checkpoint=a.checkpoint, trozo_perdida=a.trozo_perdida)
    n = sum(p.numel() for p in modelo.parametros_entrenables())
    print(f'{len(modelo.pos_largo)} capas ASRN despues de los bloques {modelo.pos_largo}: {n:,} parametros fp32')
    recorte = a.recorte if a.recorte == 'dinamico' else float(a.recorte)
    entrenar(modelo, asrn.leer_piezas(a.datos), asrn.leer_piezas(a.validacion) if a.validacion else None,
             largo=a.largo, lote=a.lote, acumular=a.acumular, pasos=a.pasos, lr=a.lr, lr_min=a.lr_min,
             calentamiento=a.calentamiento, decaimiento=a.decaimiento, recorte=recorte, evaluar_cada=a.evaluar_cada,
             lotes_eval=a.lotes_eval, guardar_en=a.guardar_en, guardar_cada=a.guardar_cada, reanudar=a.reanudar,
             semilla=a.semilla, registrar_cada=a.registrar_cada)


if __name__ == '__main__':
    principal()
