"""Presupuesto exacto de memoria (VRAM) y de cuentas (FLOPs por pieza) del injerto Qwen 2.5 + ASRN.

Python puro (no necesita torch): todo sale de la configuracion de Qwen, de las opciones de CapaASRNEstable y de como
qwen_asrn.modelo guarda tensores para la vuelta. Las cuentas de parametros son exactas (las pruebas las comparan con
los modelos de verdad); las de activaciones cuentan byte a byte lo que autograd guarda con las funciones propias de
qwen_asrn.modelo (_RMSNorm, _SwiGLU, atencion por bloques con SDPA). Lo marcado "estimado" (picos transitorios, el
contexto de CUDA, la velocidad) no se puede contar exacto sin medir en la GPU.

    python -m qwen_asrn.presupuesto                      # Qwen 2.5 7B, ventana 1024, 4 sumideros, A100 80 GB
    python -m qwen_asrn.presupuesto --largo 8192 --lote 1 --ventana 2048
"""
import argparse
import dataclasses

GIB = 2 ** 30
MIB = 2 ** 20


@dataclasses.dataclass(frozen=True)
class ConfigQwen:
    """Qwen 2.5 7B (config.json de Qwen/Qwen2.5-7B)."""
    capas: int = 28
    d: int = 3584
    ff: int = 18944
    cabezas: int = 28
    cabezas_kv: int = 4
    dim_cabeza: int = 128
    V: int = 152064
    atadas: bool = False                    # tie_word_embeddings: en el 7B, la salida es otra matriz


@dataclasses.dataclass(frozen=True)
class ConfigInjerto:
    largo_cada: int = 2
    ventana: int = 1024
    sumideros: int = 4                      # ranuras de sumidero: siempre 8 en la atencion (relleno enmascarado)
    # opciones de CapaASRNEstable (las de omision de asrn)
    n_tablas: int = 8
    n_bits: int = 14
    afin_oculto: int = 128
    chunk: int = 32
    tablas_contexto: int = 4
    confirmar: int = 16
    sufijo: int = 4
    vistos: int = 4
    nodos_racha: int = 16


RANURAS_SUMIDERO = 8


def posiciones_asrn(q, inj):
    """Como ModeloASRN: una capa ASRN despues del bloque i si i % largo_cada == 0, sin contar el ultimo."""
    return [i for i in range(q.capas - 1) if i % inj.largo_cada == 0]


# ---------------------------------------------------------------------------------------------------------------
# Parametros
# ---------------------------------------------------------------------------------------------------------------

def parametros_qwen(q):
    d, kv = q.d, q.cabezas_kv * q.dim_cabeza
    qd = q.cabezas * q.dim_cabeza
    att = (d * qd + qd) + 2 * (d * kv + kv) + qd * d          # q, k, v con sesgo; o sin sesgo
    mlp = 3 * d * q.ff
    capa = att + mlp + 2 * d
    emb = q.V * d
    total = q.capas * capa + emb + (0 if q.atadas else emb) + d
    matmul_capa = d * qd + 2 * d * kv + qd * d + mlp          # lo que multiplica (sin sesgos ni normas)
    return {'por_capa': capa, 'atencion_por_capa': att, 'mlp_por_capa': mlp, 'embeddings': emb,
            'salida': 0 if q.atadas else emb, 'total': total, 'matmul_por_capa': matmul_capa}


def parametros_capa_asrn(d, inj):
    """Exacto: los mismos parametros que CapaASRN.__init__ (asrn/nucleo.py)."""
    mitad = max(8, inj.afin_oculto // 2)
    tp, tc = inj.n_tablas - inj.tablas_contexto, inj.tablas_contexto
    afin_palabra = (d * mitad + mitad) + (mitad * tp * inj.n_bits + tp * inj.n_bits)
    afin_contexto = (2 * d * mitad + mitad) + (mitad * tc * inj.n_bits + tc * inj.n_bits)
    v = o = d * d
    g = d + 1
    g_conf = 4 + 1
    cq = ck = inj.confirmar * d
    ln = 2 * d
    resto = inj.vistos + (inj.sufijo + 1) + 2 * d + 2 + inj.n_tablas + inj.vistos + 1 + inj.vistos + inj.nodos_racha
    total = afin_palabra + afin_contexto + v + o + g + g_conf + cq + ck + ln + resto
    return {'total': total, 'v': v, 'o': o, 'hash': afin_palabra + afin_contexto, 'resto': total - v - o}


# ---------------------------------------------------------------------------------------------------------------
# Memoria de entrenamiento
# ---------------------------------------------------------------------------------------------------------------

def bytes_guardados_bloque(q, inj):
    """Bytes por pieza que autograd guarda en un bloque de Qwen congelado (pesos sin gradiente: un Linear guarda solo
    su peso, nada por pieza). bf16 = 2 bytes. Ver qwen_asrn.modelo."""
    d, qd, kv = q.d, q.cabezas * q.dim_cabeza, q.cabezas_kv * q.dim_cabeza
    W = inj.ventana
    llaves = (2 * W + RANURAS_SUMIDERO) / W                   # por bloque de W preguntas: 2W + 8 llaves
    partes = {
        'rmsnorm x2 (entrada bf16)': 2 * 2 * d,
        'q (SDPA)': 2 * qd,
        'k, v por bloques (SDPA)': 2 * 2 * kv * llaves,
        'salida de la atencion (SDPA)': 2 * qd,
        'logsumexp fp32 (SDPA)': 4 * q.cabezas,
        'SwiGLU (gate y up, bf16)': 2 * 2 * q.ff,
    }
    return partes


def bytes_mascara(q, inj):
    """La mascara aditiva por bloques (bf16), una sola para todas las capas: G*W filas x (2W+8) llaves por bloque de
    W preguntas."""
    G = q.cabezas // q.cabezas_kv
    return 2 * G * (2 * inj.ventana + RANURAS_SUMIDERO)


def activaciones_por_pieza(q, inj, checkpoint='asrn'):
    """Bytes guardados para la vuelta por pieza del lote (B*L piezas)."""
    pos = posiciones_asrn(q, inj)
    n_asrn = len(pos)
    bloques_con_vuelta = q.capas - (pos[0] + 1)               # los bloques antes de la primera ASRN no la necesitan
    bloque = sum(bytes_guardados_bloque(q, inj).values())
    h_bf16 = 2 * q.d
    if checkpoint == 'todo':
        bloque_total = bloques_con_vuelta * h_bf16
    else:
        bloque_total = bloques_con_vuelta * bloque
    asrn = n_asrn * h_bf16 + h_bf16 + 8                       # h de entrada de cada capa, E (una vez) y x (int64)
    if checkpoint == 'no':
        asrn = None                                           # no se cuenta: ver el documento
    final = h_bf16 + 8                                        # entrada de la norma final y las etiquetas
    return {'bloques': bloque_total, 'asrn': asrn, 'final': final, 'mascara': bytes_mascara(q, inj),
            'total': bloque_total + (asrn or 0) + final + bytes_mascara(q, inj), 'bloques_con_vuelta': bloques_con_vuelta,
            'por_bloque': bloque}


def memoria_entrenamiento(q, inj, piezas, checkpoint='asrn', trozo_perdida=1024, contexto_cuda=3 * GIB,
                          transitorio_asrn=256 * 1024, transitorio_bloque=160 * 1024):
    pq = parametros_qwen(q)
    pa = parametros_capa_asrn(q.d, inj)['total'] * len(posiciones_asrn(q, inj))
    act = activaciones_por_pieza(q, inj, checkpoint)
    perdida = trozo_perdida * q.V * (2 + 4 + 4)               # logits bf16 + fp32 + su gradiente fp32 (un trozo)
    transitorio = max(transitorio_asrn, transitorio_bloque if checkpoint == 'todo' else 0) * piezas
    r = {
        'pesos Qwen bf16': 2 * pq['total'],
        'pesos ASRN fp32': 4 * pa,
        'gradientes ASRN fp32': 4 * pa,
        'AdamW m y v fp32 (solo ASRN)': 8 * pa,
        'activaciones guardadas': act['total'] * piezas,
        'pico transitorio de una capa (estimado)': transitorio,
        'perdida por trozos (un trozo)': perdida,
        'contexto CUDA + fragmentacion (estimado)': contexto_cuda,
    }
    r['TOTAL'] = sum(r.values())
    return r


def piezas_maximas(q, inj, memoria_gpu, checkpoint='asrn', **kw):
    fijo = sum(v for k, v in memoria_entrenamiento(q, inj, 0, checkpoint, **kw).items() if k != 'TOTAL')
    por_pieza = memoria_entrenamiento(q, inj, 1, checkpoint, **kw)['TOTAL'] - fijo
    return int((memoria_gpu - fijo) // por_pieza)


# ---------------------------------------------------------------------------------------------------------------
# FLOPs por pieza
# ---------------------------------------------------------------------------------------------------------------

def flops_asrn_capa(d, inj, escritura=False):
    T, C, P, cf = inj.n_tablas, inj.chunk, inj.n_tablas * inj.n_bits, inj.confirmar
    pa = parametros_capa_asrn(d, inj)
    hash_ = 2 * pa['hash']
    proy = 2 * (2 * cf + 2) * d + 2 * d                       # cq, ck, reloj y la puerta g
    if escritura:                                             # T*C candidatos; v factorizado: una sola d x d
        busqueda = T * C * (2 * (P + cf) + 2 * d)
        return {'v y o': 2 * 2 * d * d, 'hash': hash_, 'proyecciones': proy, 'busqueda': busqueda,
                'total': 4 * d * d + hash_ + proy + busqueda}
    llaves = 2 * C                                            # cada bloque de C preguntas contra 2C llaves
    busqueda = T * llaves * (2 * (P + cf) + 2 * d)
    return {'v y o': 4 * d * d, 'hash': hash_, 'proyecciones': proy, 'busqueda': busqueda,
            'total': 4 * d * d + hash_ + proy + busqueda}


def flops_por_pieza(q, inj, checkpoint='asrn'):
    pq = parametros_qwen(q)
    pos = posiciones_asrn(q, inj)
    qd = q.cabezas * q.dim_cabeza
    lin = 2 * pq['matmul_por_capa']
    head = 2 * pq['embeddings']
    att_util = 4 * inj.ventana * qd                           # QK^T y AV con W llaves
    att_calc = 4 * (2 * inj.ventana + RANURAS_SUMIDERO) * qd  # lo que calcula la atencion por bloques
    asrn = flops_asrn_capa(q.d, inj)['total']
    ida = q.capas * (lin + att_calc) + head + len(pos) * asrn
    con_vuelta = q.capas - (pos[0] + 1)
    vuelta = (con_vuelta * lin + head                         # congelado: solo dX (sin dW)
              + con_vuelta * 2.5 * att_calc                   # atencion: dQ, dK, dV y recalcular P
              + len(pos) * 2 * asrn)                          # ASRN: dX y dW
    recalculo = len(pos) * asrn + head                        # checkpoint='asrn' y la perdida por trozos
    if checkpoint == 'todo':
        recalculo += con_vuelta * (lin + att_calc)
    return {'ida': ida, 'ida_util': q.capas * (lin + att_util) + head + len(pos) * asrn,
            'vuelta': vuelta, 'recalculo': recalculo, 'entrenamiento': ida + vuelta + recalculo,
            'qwen_lineal': q.capas * lin + head, 'atencion': q.capas * att_calc, 'asrn': len(pos) * asrn,
            'asrn_capa': asrn}


def flops_escritura(q, inj):
    pq = parametros_qwen(q)
    qd = q.cabezas * q.dim_cabeza
    lin = q.capas * 2 * pq['matmul_por_capa'] + 2 * pq['embeddings']
    att = q.capas * 4 * (inj.ventana + RANURAS_SUMIDERO) * qd
    asrn = len(posiciones_asrn(q, inj)) * flops_asrn_capa(q.d, inj, escritura=True)['total']
    return {'qwen_lineal': lin, 'atencion': att, 'asrn': asrn, 'total': lin + att + asrn}


# ---------------------------------------------------------------------------------------------------------------
# Escritura O(1): estado por texto
# ---------------------------------------------------------------------------------------------------------------

def memoria_escritura(q, inj, contexto, v_factorizado=True):
    """Bytes por texto del lote con `contexto` piezas ya vistas."""
    n_asrn = len(posiciones_asrn(q, inj))
    kv = q.capas * (inj.ventana + RANURAS_SUMIDERO) * 2 * q.cabezas_kv * q.dim_cabeza * 2
    anillos = n_asrn * inj.n_tablas * 2 ** inj.n_bits * (4 * inj.chunk + 8)     # _Escritor._bytes_anillos
    P = inj.n_tablas * inj.n_bits
    por_pieza_capa = 4 * P + 4 * inj.n_tablas + 4 * inj.confirmar + 4 * inj.sufijo + 8 + 4   # phi ids Kc Pf B log_alfa
    if not v_factorizado:
        por_pieza_capa += 4 * q.d                             # V = v(E_{s+1}) en fp32
    hawkes = n_asrn * inj.vistos * (8 + 4)                    # B_u (64 bits) y A por clave nueva (a lo sumo)
    por_pieza = n_asrn * por_pieza_capa + 8 + 4 + hawkes      # + el texto (int64) y las firmas (fp32)
    return {'cache KV (ventana + sumideros)': kv, 'anillos de casilleros': anillos,
            'memoria por pieza': por_pieza, 'memoria del contexto': por_pieza * contexto,
            'TOTAL': kv + anillos + por_pieza * contexto, 'por_pieza_capa': por_pieza_capa}


# ---------------------------------------------------------------------------------------------------------------
# Opcion B: cambiar la atencion de los bloques impares por una convolucion causal k=4
# ---------------------------------------------------------------------------------------------------------------

def opcion_b(q, inj, k=4):
    pq = parametros_qwen(q)
    impares = [i for i in range(q.capas) if i % inj.largo_cada == 1]
    qd = q.cabezas * q.dim_cabeza
    att_lin = 2 * (pq['matmul_por_capa'] - 3 * q.d * q.ff)
    att_core = 4 * (2 * inj.ventana + RANURAS_SUMIDERO) * qd
    densa = {'parametros': q.d * q.d * k + q.d, 'flops': 2 * q.d * q.d * k}
    canal = {'parametros': q.d * k + q.d, 'flops': 2 * q.d * k}
    r = {'bloques': len(impares), 'flops_quitados': len(impares) * (att_lin + att_core)}
    for nombre, c in (('densa', densa), ('por_canal', canal)):
        r[nombre] = {'parametros_nuevos': len(impares) * c['parametros'],
                     'bytes_entrenables': 16 * len(impares) * c['parametros'],   # fp32 + grad + m + v
                     'flops_agregados': len(impares) * c['flops'],
                     'flops_netos': len(impares) * c['flops'] - r['flops_quitados']}
    r['pesos_qwen_quitados_bytes'] = 2 * len(impares) * pq['atencion_por_capa']
    r['kv_ahorrado_bytes'] = len(impares) * (inj.ventana + RANURAS_SUMIDERO) * 2 * q.cabezas_kv * q.dim_cabeza * 2
    return r


# ---------------------------------------------------------------------------------------------------------------

def _g(x):
    return f'{x / GIB:,.2f} GiB ({x / 1e9:,.2f} GB)'


def informe(q=ConfigQwen(), inj=ConfigInjerto(), largo=4096, lote=1, memoria_gpu=80 * GIB, tflops=312e12, mfu=0.40,
            contexto=1_000_000):
    out = []
    p = out.append
    pq = parametros_qwen(q)
    pa = parametros_capa_asrn(q.d, inj)
    pos = posiciones_asrn(q, inj)
    p(f'## Parametros\n')
    p(f'- Qwen: {pq["total"]:,} ({pq["por_capa"]:,} por bloque; embeddings {pq["embeddings"]:,}; salida '
      f'{pq["salida"]:,})')
    p(f'- CapaASRNEstable(d={q.d}): {pa["total"]:,} (v {pa["v"]:,} + o {pa["o"]:,} + hash {pa["hash"]:,} + resto '
      f'{pa["resto"] - pa["hash"]:,})')
    p(f'- {len(pos)} capas ASRN despues de los bloques {pos}: {len(pos) * pa["total"]:,} entrenables\n')
    piezas = largo * lote
    for ck in ('asrn', 'todo'):
        act = activaciones_por_pieza(q, inj, ck)
        p(f'## VRAM de entrenamiento, checkpoint={ck!r}, lote {lote} x largo {largo} = {piezas:,} piezas\n')
        if ck == 'asrn':
            p('Por pieza y por bloque de Qwen con vuelta (bytes):\n')
            for k, v in bytes_guardados_bloque(q, inj).items():
                p(f'  - {k}: {v:,.0f}')
            p(f'  - bloque: {act["por_bloque"]:,.0f} x {act["bloques_con_vuelta"]} bloques\n')
        p(f'Activaciones por pieza: {act["total"]:,.0f} B ({act["total"] / MIB:.3f} MiB): bloques '
          f'{act["bloques"]:,.0f} + ASRN {act["asrn"]:,.0f} + final {act["final"]:,} + mascara {act["mascara"]:,}\n')
        p('| parte | memoria |\n|---|---|')
        for k, v in memoria_entrenamiento(q, inj, piezas, ck).items():
            p(f'| {k} | {_g(v)} |')
        p(f'\nMaximo de piezas por paso en {memoria_gpu / GIB:.0f} GiB: '
          f'{piezas_maximas(q, inj, memoria_gpu, ck):,}\n')
    f = flops_por_pieza(q, inj)
    ft = flops_por_pieza(q, inj, 'todo')
    p('## FLOPs por pieza (entrenamiento)\n')
    p(f'- ida: {f["ida"] / 1e9:.3f} GFLOP (Qwen lineal {f["qwen_lineal"] / 1e9:.3f}, atencion por bloques '
      f'{f["atencion"] / 1e9:.3f}, ASRN {f["asrn"] / 1e9:.3f} = {len(pos)} x {f["asrn_capa"] / 1e6:.2f} M); util '
      f'(W llaves) {f["ida_util"] / 1e9:.3f}')
    p(f'- vuelta: {f["vuelta"] / 1e9:.3f} GFLOP; recalculo: {f["recalculo"] / 1e9:.3f}')
    p(f'- paso completo: {f["entrenamiento"] / 1e9:.3f} GFLOP/pieza (checkpoint todo: '
      f'{ft["entrenamiento"] / 1e9:.3f})')
    for nombre, x in (('asrn', f), ('todo', ft)):
        p(f'- A100 ({tflops / 1e12:.0f} TFLOPS bf16, MFU {mfu:.0%}, estimado) checkpoint={nombre}: '
          f'{tflops * mfu / x["entrenamiento"]:,.0f} piezas/s')
    fe = flops_escritura(q, inj)
    p(f'\n## Escritura O(1)\n\n- FLOPs por pieza: {fe["total"] / 1e9:.3f} GFLOP (Qwen {fe["qwen_lineal"] / 1e9:.3f}, '
      f'atencion {fe["atencion"] / 1e9:.3f}, ASRN {fe["asrn"] / 1e9:.3f})')
    leer = 2 * pq['total'] + 4 * len(pos) * pa['total']
    p(f'- bytes de pesos leidos por paso: {_g(leer)}; a 2.039 TB/s: {leer / 2.039e12 * 1e3:.2f} ms por paso '
      f'(<= {2.039e12 / leer:.0f} pasos/s; cada paso escribe una pieza por texto del lote)')
    for fact in (True, False):
        m = memoria_escritura(q, inj, contexto, fact)
        p(f'\nv_factorizado={fact}, contexto {contexto:,} piezas, por texto:\n')
        for k, v in m.items():
            if k == 'por_pieza_capa':
                continue
            p(f'  - {k}: {v:,} B' if k == 'memoria por pieza' else f'  - {k}: {_g(v)}')
    b = opcion_b(q, inj)
    p(f'\n## Opcion B ({b["bloques"]} bloques sin atencion)\n')
    p(f'- quita {b["flops_quitados"] / 1e9:.3f} GFLOP/pieza de atencion; libera {_g(b["pesos_qwen_quitados_bytes"])} '
      f'de pesos y {b["kv_ahorrado_bytes"] / MIB:.1f} MiB de cache KV por texto')
    for nombre in ('densa', 'por_canal'):
        r = b[nombre]
        p(f'- convolucion {nombre}: +{r["parametros_nuevos"]:,} parametros entrenables ({_g(r["bytes_entrenables"])} '
          f'con AdamW), FLOPs netos {r["flops_netos"] / 1e9:+.3f} GFLOP/pieza')
    return '\n'.join(out)


def main(args=None):
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--largo', type=int, default=4096)
    a.add_argument('--lote', type=int, default=1)
    a.add_argument('--ventana', type=int, default=1024)
    a.add_argument('--sumideros', type=int, default=4)
    a.add_argument('--memoria_gib', type=float, default=80.0)
    a.add_argument('--contexto', type=int, default=1_000_000)
    a = a.parse_args(args)
    inj = ConfigInjerto(ventana=a.ventana, sumideros=a.sumideros)
    print(informe(inj=inj, largo=a.largo, lote=a.lote, memoria_gpu=a.memoria_gib * GIB, contexto=a.contexto))


if __name__ == '__main__':
    main()
