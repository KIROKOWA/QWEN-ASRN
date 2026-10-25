# okc: problem=muse_attention_decode
"""Atencion con puerta de Muse Spark Open (un paso de decodificacion o un bloque de verificacion de 8 piezas) en UN
solo kernel de Triton, en vez de los cinco de SGLang (FusedQKNorm, store_cache, q a FP8, FA3 y rms_norm_gate).

v8 junta los dos metodos medidos y usa en cada forma el que gano en el juez (ver kernel_fn al final): el de
quorum de la v6/v7, descrito aqui, y el de pedazos de la v2 (dos kernels, softmax en linea por pedazo).

v6, "la atencion como funcion de particion". Ideas tomadas de otras areas, cada una porque resuelve algo concreto:

1. Fisica estadistica: energia de referencia fija. El softmax es una funcion de particion, Z = sum_i e^(s_i), y como
   en fisica cada estado se puede medir contra cualquier energia de referencia c sin cambiar el resultado. FlashAttention
   usa como referencia el maximo corriendo, que obliga a calcular el maximo de cada bloque y a reescalar lo acumulado
   cada vez que sube. Aqui c se conoce antes de empezar, por la desigualdad de Cauchy-Schwarz: Muse normaliza consultas
   y llaves (QK-norm), asi que |q.k| <= |q| |k| = (qk_scale_factor * 8) * 8, y cada logit (en base 2) cae en un
   intervalo de ancho ~126, que cabe en el rango normal de float32 (2^-126 .. 2^127). Con c = techo - 40 todos los
   terminos quedan en [2^-86, 2^40]: sin maximo, sin reescalar y sin desbordes, con llaves normalizadas y con holgura
   para llaves de norma hasta ~1.4 veces la normal (un tope en 2^100 evita inf con datos absurdos).
2. Cantidades extensivas: con la misma c en todos los programas, Z y el acumulado sum_i e^(s_i - c) v_i se suman como
   la energia o la masa: cada programa agrega lo suyo con atomicos a un acumulador global. No hay pedazos que guardar,
   ni maximos que conciliar, ni un segundo kernel que los junte (en la v2, en verify, los pedazos movian 4 veces mas
   bytes que el propio cache).
3. Biologia, quorum sensing: las bacterias cuentan moleculas senal y actuan en grupo al llegar a un umbral. Cada
   programa, al terminar, suma 1 a un contador de su (peticion, cabeza kv, grupo de filas); el que completa el quorum,
   el ultimo en llegar, cierra: divide, normaliza, aplica la puerta y escribe. Lee los acumuladores con un intercambio
   atomico que los deja en cero, y vuelve a poner el contador en cero: el estado queda limpio para la siguiente llamada
   sin un memset. Un solo lanzamiento: en las capas de ventana (46 de cada 62 llamadas) el costo es casi todo latencia.
4. Teoria de colas (v4): el total de llaves se parte en trabajos del mismo tamano segun las longitudes reales, leidas
   en la GPU, tantos como lugares tiene la maquina.
5. Organizador de memoria (v5): anillo asincrono de varias etapas en memoria compartida para las filas sueltas del
   cache. Ademas, los programas de un mismo trabajo (sus dos cabezas kv y sus grupos de filas) son vecinos en la malla,
   asi que piden las mismas filas de 128 bytes casi al mismo tiempo: la memoria las trae una vez y el resto sale del L2.
   Y, como en las bases de datos que esconden la latencia de sondeos sueltos (prefetching por grupos, AMAC), un programa
   por trabajo pide al L2 las filas de vueltas futuras sin gastar registros ni memoria compartida.

_trabajo solo lee posiciones anteriores al bloque y el trabajo 0 de cada peticion escribe las del bloque: no hay
carrera en el cache. Sin sincronizar con la CPU y sin ramas del lado de la CPU sobre valores: se captura en un CUDA
graph. Los acumuladores persistentes no guardan nada de los datos entre llamadas: empiezan y terminan en cero."""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64]: q, keys, values, gate logits
    positions: torch.Tensor    # int64 [B*L], each new token's position (seq_lens[b] + i)
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row (page * page_size + offset)
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # FA3's tile schedule; not used here
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


LOG2E = 1.4426950408889634
ROPE_POSITIONS = 1 << 21      # como el RotaryEmbedding de SGLang (y baseline.py)
HOLGURA = 40.0                # la energia de referencia queda 40 (en base 2) bajo el techo de Cauchy-Schwarz
TOPE = tl.constexpr(100.0)    # ningun termino pasa de 2^100: solo actua con llaves de norma absurda (evita inf)
# Por forma (q_len, contexto completo): (llaves por vuelta, etapas del anillo, warps, programas por SM, grupos de filas,
# vueltas de adelanto del prefetch al L2, consultas en FP8, tandas de trabajos, llaves minimas por trabajo). Medida con
# compilar_sm90.py (ptxas -v para sm_90): decode 128 registros -> 4 por SM (en bf16: FP8 no ahorra nada alli); verify
# con las 64 filas juntas y consultas en FP8 usa wgmma con las llaves directo del anillo: 128-164 registros -> 3-4 por
# SM (en bf16, 174 -> 2 por SM). En verify cada trabajo suma 64 x 64 valores con atomicos: trabajos mas grandes.
# v7: el reparto de la v2 (la mejor medida en contexto completo): ~16 programas por SM, cortos, en vez de 1-2 tandas
# largas (v4-v6, 30-40 % mas lentas alli); sin prefetch al L2 ni evict_first (cada fila de 128 bytes la leen dos
# programas, uno por cabeza kv: evict_first la sacaba del L2 antes de que llegara el segundo).
CONFIG = {(1, True): (128, 3, 4, 16, 1, 0, False, 1, 256), (1, False): (64, 2, 4, 16, 1, 0, False, 1, 64),
          (8, True): (64, 4, 4, 16, 1, 0, True, 1, 256), (8, False): (64, 3, 4, 16, 1, 0, True, 1, 64)}
_ROPE = {}
_SMS = {}
_ESTADO = {}


def _cos_sin(theta, d, device):
    """[2^21, d] float32 cos | sin, como lo arma SGLang, guardado por theta (no depende de las entradas)."""
    clave = (theta, d, str(device))
    tabla = _ROPE.get(clave)
    if tabla is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        tabla = _ROPE[clave] = torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()
    return tabla


def _n_sms(device):
    if device.type != "cuda":
        return 132
    if device not in _SMS:
        _SMS[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SMS[device]


def _estado(device, n_filas, n_grupos):
    """Acumuladores (sum e^(s-c) v, sum e^(s-c)) y contadores del quorum, en cero. Se crean una vez por forma y el
    programa que cierra cada grupo los deja otra vez en cero: no guardan nada de una llamada a la otra."""
    clave = (str(device), n_filas, n_grupos)
    estado = _ESTADO.get(clave)
    if estado is None:
        estado = _ESTADO[clave] = (torch.zeros((n_filas, 64), dtype=torch.float32, device=device),
                                   torch.zeros((n_filas,), dtype=torch.float32, device=device),
                                   torch.zeros((n_grupos,), dtype=torch.int32, device=device))
    return estado


@triton.jit
def _a_fp8(x):
    """x float32 a E4M3, redondeando al par mas cercano. La mantisa se redondea antes a 3 bits con enteros, asi el
    cambio de tipo ya recibe un valor exacto: lo mismo que la conversion de Hopper, y correcto tambien en el
    interprete de Triton (cuya conversion no sube el exponente cuando el redondeo desborda: 1.97 -> 1.0)."""
    bits = x.to(tl.int32, bitcast=True)
    bits = (bits + 0x7FFFF + ((bits >> 20) & 1)) & -1048576  # -1048576 = ~0xFFFFF
    return bits.to(tl.float32, bitcast=True).to(tl.float8e4nv)


@triton.jit
def _norma_rope(base, pos, ROPE, escala, CON_ROPE: tl.constexpr):
    """Las filas de 64 que empiezan en base ([R] punteros): RMSNorm sin escala, por escala, y RoPE intercalado (pares
    2i, 2i+1) en la posicion pos ([R]) si CON_ROPE. Devuelve [R, 64] float32 en el orden original."""
    mitad = tl.arange(0, 32)
    par = tl.load(base[:, None] + 2 * mitad[None, :]).to(tl.float32)
    impar = tl.load(base[:, None] + 2 * mitad[None, :] + 1).to(tl.float32)
    inv = tl.rsqrt((tl.sum(par * par, 1) + tl.sum(impar * impar, 1)) / 64 + 1e-5) * escala
    par, impar = par * inv[:, None], impar * inv[:, None]
    if CON_ROPE:
        fila = ROPE + pos.to(tl.int64)[:, None] * 64 + mitad[None, :]
        c, s = tl.load(fila), tl.load(fila + 32)
        par, impar = par * c - impar * s, par * s + impar * c
    return tl.reshape(tl.join(par, impar), (par.shape[0], 64))


@triton.jit
def _filas(TABLA, b, paso_tabla, t, fin, PAGINA: tl.constexpr):
    """Filas del cache (int64) de las posiciones t de la peticion b; 0 en las que no llegan a fin."""
    ok = t < fin
    if PAGINA == 1:
        return tl.load(TABLA + b * paso_tabla + t, mask=ok, other=0).to(tl.int64)
    return tl.load(TABLA + b * paso_tabla + t // PAGINA, mask=ok, other=0).to(tl.int64) * PAGINA + t % PAGINA


@triton.jit
def _prefetch_l2(punteros):
    """Pide al L2 las lineas de estos punteros, sin destino: no ocupa registros ni memoria compartida."""
    tl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.u32 $0, 0;", "=r,l", [punteros], dtype=tl.int32,
                              is_pure=False, pack=1)


@triton.jit
def _reparto(SEQ, B: tl.constexpr, BB: tl.constexpr, VENTANA: tl.constexpr, n_trabajos, min_chunk,
             BLOQUE_N: tl.constexpr):
    """El plan de trabajos, igual en todos los programas: las llaves ya guardadas que ve cada peticion (lens), el
    tamano de trabajo (chunk: el total entre n_trabajos, en vueltas enteras, al menos min_chunk), cuantos trabajos
    le tocan a cada una (n, al menos uno: alguien tiene que cerrar) y donde terminan en la lista (acum)."""
    c = tl.arange(0, BB)
    es = c < B
    seqs = tl.load(SEQ + c, mask=es, other=0)
    if VENTANA > 0:
        lens = tl.minimum(seqs, VENTANA - 1)
    else:
        lens = seqs
    total = tl.sum(lens, 0)
    chunk = tl.maximum(((total + n_trabajos - 1) // n_trabajos + BLOQUE_N - 1) // BLOQUE_N * BLOQUE_N, min_chunk)
    n = tl.where(es, tl.maximum((lens + chunk - 1) // chunk, 1), 0)
    return c, seqs, lens, chunk, n, tl.cumsum(n, 0)


@triton.jit
def _sumar(q, k, v, valido, suma, acc, escala, c_ref, QK_FP8: tl.constexpr, EN_CPU: tl.constexpr):
    """Un bloque de llaves k y valores v ([N, 64], E4M3) sobre las consultas q: sin maximo ni reescalado, cada termino
    es 2^(s - c) con la energia de referencia c fija."""
    if EN_CPU:  # el interprete de Triton multiplica mal en bf16 y fp8: alli, en float32 exacto
        sc = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32)), input_precision="ieee")
    elif QK_FP8:
        sc = tl.dot(q, tl.trans(k))
    else:
        sc = tl.dot(q, tl.trans(k.to(tl.bfloat16)))
    p = tl.math.exp2(tl.where(valido, tl.minimum(sc * escala - c_ref, TOPE), float("-inf")))
    suma = suma + tl.sum(p, 1)
    if EN_CPU:
        acc = acc + tl.dot(p.to(tl.bfloat16).to(tl.float32), v.to(tl.float32), input_precision="ieee")
    else:
        acc = acc + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
    return suma, acc


@triton.jit
def _cerrar(QKV, ACC, SUMA, VS, OUT, donde, fila_ok, fila_tok, salida, cabeza, paso_qkv, HQ: tl.constexpr,
            HKV: tl.constexpr, FILAS: tl.constexpr):
    """El cierre del grupo: lee (y deja en cero) los acumuladores, divide, RMSNorm por cabeza, por sigmoid(gate)."""
    d = tl.arange(0, 64)
    # Lectura coherente en el L2 (sumar 0 de forma atomica) y despues ceros: ya nadie mas suma a este grupo.
    p_acc = ACC + donde[:, None] * 64 + d[None, :]
    acc = tl.atomic_add(p_acc, tl.zeros([FILAS, 64], tl.float32), mask=fila_ok[:, None], sem="relaxed")
    suma = tl.atomic_add(SUMA + donde, tl.zeros([FILAS], tl.float32), mask=fila_ok, sem="relaxed")
    tl.store(p_acc, tl.zeros([FILAS, 64], tl.float32), mask=fila_ok[:, None])
    tl.store(SUMA + donde, tl.zeros([FILAS], tl.float32), mask=fila_ok)
    acc = tl.where(fila_ok[:, None], acc, 0.0)  # las filas de relleno no se leyeron: nada indefinido sigue adelante
    o = acc / tl.where(fila_ok, suma, 1.0)[:, None] * tl.load(VS)
    o = o * tl.rsqrt(tl.sum(o * o, 1) / 64 + 1e-5)[:, None]
    puerta = tl.load(QKV + fila_tok[:, None] + (HQ + 2 * HKV + cabeza)[:, None] * 64 + d[None, :],
                     mask=fila_ok[:, None], other=0.0).to(tl.float32)
    o = o * tl.sigmoid(puerta)
    tl.store(OUT + salida[:, None] + cabeza[:, None] * 64 + d[None, :], o.to(tl.bfloat16), mask=fila_ok[:, None])


@triton.jit
def _atencion(QKV, SEQ, LOC, TABLA, KC, VC, KS, VS, ROPE, ACC, SUMA, CNT, OUT, paso_qkv, paso_tabla, n_trabajos,
              min_chunk, qk_escala, escala_log2, c_ref, B: tl.constexpr, BB: tl.constexpr, HQ: tl.constexpr,
              HKV: tl.constexpr, L: tl.constexpr, G: tl.constexpr, R: tl.constexpr, FILAS: tl.constexpr,
              PAGINA: tl.constexpr, VENTANA: tl.constexpr, CON_ROPE: tl.constexpr, BLOQUE_N: tl.constexpr,
              ETAPAS: tl.constexpr, BLOQUE_L: tl.constexpr, PREFETCH: tl.constexpr, QK_FP8: tl.constexpr,
              QUORUM: tl.constexpr, EN_CPU: tl.constexpr):
    """Un trabajo (program_id 1) para una cabeza kv y un grupo de filas de consulta (program_id 0, el grupo de filas
    el que mas rapido cambia: los programas de un mismo trabajo son vecinos y leen las mismas filas a la vez)."""
    hr = tl.program_id(0)
    w = tl.program_id(1)
    r = hr % R
    h = hr // R
    c, seqs, lens, chunk, n, acum = _reparto(SEQ, B, BB, VENTANA, n_trabajos, min_chunk, BLOQUE_N)
    b = tl.sum((acum <= w).to(tl.int32), 0)  # la peticion de este trabajo; B si sobra
    hay = b < B
    b = tl.minimum(b, B - 1)
    es_b = c == b
    seq = tl.sum(tl.where(es_b, seqs, 0), 0)
    n_b = tl.sum(tl.where(es_b, n, 0), 0)
    s = w - tl.sum(tl.where(es_b, acum - n, 0), 0)  # cual de los trabajos de su peticion
    inicio = seq - tl.sum(tl.where(es_b, lens, 0), 0) + s * chunk
    fin = tl.where(hay, tl.minimum(inicio + chunk, seq), inicio)

    # Las filas de consulta de este grupo: fila_global = r * (L * G / R) + i, de la pieza j y la cabeza h * G + g.
    filas = tl.arange(0, FILAS)
    fila_ok = filas < L * G // R
    gr = tl.where(fila_ok, r * (L * G // R) + filas, 0)
    j = gr // G
    cabeza = h * G + gr % G
    pos = seq + j
    d = tl.arange(0, 64)
    fila_tok = (b * L + j).to(tl.int64) * paso_qkv
    q = _norma_rope(QKV + fila_tok + cabeza * 64, pos, ROPE, 1.0, CON_ROPE)
    if QK_FP8:  # como FA3: la consulta normalizada (antes de qk_scale_factor) en E4M3; el factor va en la escala
        q = _a_fp8(q)
        escala = tl.load(KS) * escala_log2 * qk_escala
    else:
        q = (q * qk_escala).to(tl.bfloat16)
        escala = tl.load(KS) * escala_log2

    suma = tl.zeros([FILAS], tl.float32)
    acc = tl.zeros([FILAS, 64], tl.float32)
    col = h * 64 + d[None, :]
    fila_sig = _filas(TABLA, b, paso_tabla, inicio + tl.arange(0, BLOQUE_N), fin, PAGINA)
    for t0 in tl.range(inicio, fin, BLOQUE_N, num_stages=ETAPAS):
        t = t0 + tl.arange(0, BLOQUE_N)
        ok = t < fin
        off = fila_sig[:, None] * (HKV * 64) + col
        fila_sig = _filas(TABLA, b, paso_tabla, t + BLOQUE_N, fin, PAGINA)
        if PREFETCH > 0:
            if not EN_CPU:  # todos piden lo mismo: el L2 junta los pedidos repetidos a una misma linea
                adelante = _filas(TABLA, b, paso_tabla, t + PREFETCH * BLOQUE_N, fin, PAGINA) * (HKV * 64)
                _prefetch_l2(KC + adelante)
                _prefetch_l2(VC + adelante)
        k = tl.load(KC + off, mask=ok[:, None], other=0.0)
        v = tl.load(VC + off, mask=ok[:, None], other=0.0)
        valido = ok[None, :]
        if VENTANA > 0:
            valido = valido & (t[None, :] > pos[:, None] - VENTANA)
        suma, acc = _sumar(q, k, v, valido, suma, acc, escala, c_ref, QK_FP8, EN_CPU)

    # Las L piezas nuevas del bloque, en el primer trabajo de la peticion: cada fila j ve las i <= j (siempre dentro de
    # la ventana). Se normalizan, rotan y redondean a E4M3 como las guardadas; el grupo 0 las escribe en el cache.
    if (s == 0) & hay:
        li = tl.arange(0, BLOQUE_L)
        hay_l = li < L
        i_c = tl.where(hay_l, li, 0)
        tok = (b * L + i_c).to(tl.int64)
        ks = tl.load(KS)
        vs = tl.load(VS)
        k8 = _a_fp8(_norma_rope(QKV + tok * paso_qkv + (HQ + h) * 64, seq + i_c, ROPE, 1.0, CON_ROPE) / ks)
        v8 = _a_fp8(tl.load(QKV + tok[:, None] * paso_qkv + (HQ + HKV + h) * 64 + d[None, :]).to(tl.float32) / vs)
        destino = tl.load(LOC + tok)[:, None] * (HKV * 64) + h * 64 + d[None, :]
        escribe = hay_l[:, None] & (r == 0)
        tl.store(KC + destino, k8, mask=escribe)
        tl.store(VC + destino, v8, mask=escribe)
        valido = hay_l[None, :] & (li[None, :] <= j[:, None])
        suma, acc = _sumar(q, k8, v8, valido, suma, acc, escala, c_ref, QK_FP8, EN_CPU)

    # Cantidades extensivas: lo de este trabajo se suma al acumulador de su grupo.
    grupo = (b * HKV + h) * R + r
    donde = grupo * FILAS + filas
    guarda = fila_ok & hay
    tl.atomic_add(ACC + donde[:, None] * 64 + d[None, :], acc, mask=guarda[:, None], sem="relaxed")
    tl.atomic_add(SUMA + donde, suma, mask=guarda, sem="relaxed")
    if QUORUM:
        tl.debug_barrier()  # todos los hilos sumaron antes de que uno avise (como la sincronizacion de malla de CUDA)
        if hay:
            llegada = tl.atomic_add(CNT + grupo, 1, sem="acq_rel")
            if llegada == n_b - 1:  # el ultimo en llegar cierra y deja el contador en cero
                tl.store(CNT + grupo, 0)
                _cerrar(QKV, ACC, SUMA, VS, OUT, donde, fila_ok, fila_tok, (b * L + j).to(tl.int64) * (HQ * 64),
                        cabeza, paso_qkv, HQ, HKV, FILAS)


@triton.jit
def _cierre(QKV, ACC, SUMA, VS, OUT, paso_qkv, HQ: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr,
            G: tl.constexpr, R: tl.constexpr, FILAS: tl.constexpr):
    """Sin quorum (QUORUM=False): el cierre de cada grupo en un segundo kernel, despues de todos los trabajos."""
    grupo = tl.program_id(0)
    r = grupo % R
    h = (grupo // R) % HKV
    b = grupo // (R * HKV)
    filas = tl.arange(0, FILAS)
    fila_ok = filas < L * G // R
    gr = tl.where(fila_ok, r * (L * G // R) + filas, 0)
    j = gr // G
    _cerrar(QKV, ACC, SUMA, VS, OUT, grupo * FILAS + filas, fila_ok, (b * L + j).to(tl.int64) * paso_qkv,
            (b * L + j).to(tl.int64) * (HQ * 64), h * G + gr % G, paso_qkv, HQ, HKV, FILAS)


QUORUM = True



# --- Metodo de pedazos (v2): dos kernels, softmax en linea por pedazo y un segundo kernel que los junta ---------------
VACIO = tl.constexpr(-1e30)   # maximo de una fila sin llaves validas: finito, para que 2^(m - m') no de NaN
PROGRAMAS_POR_SM = 16         # _parcial_v2: muchos programas en vuelo, para tapar la espera de las lecturas sueltas
BLOQUE_S_V2 = 64              # _final_v2 junta los pedazos de a tantos


@triton.jit
def _parcial_v2(QKV, SEQ, TABLA, KC, VC, KS, ROPE, MP, LP, AP, paso_qkv, paso_tabla, chunk, n_pedazos, qk_escala,
             escala_log2, HQ: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr, G: tl.constexpr,
             PAGINA: tl.constexpr, VENTANA: tl.constexpr, CON_ROPE: tl.constexpr, BLOQUE_M: tl.constexpr,
             BLOQUE_N: tl.constexpr, EN_CPU: tl.constexpr):
    s = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // HKV
    h = bh % HKV
    seq = tl.load(SEQ + b)
    filas = tl.arange(0, BLOQUE_M)
    fila_ok = filas < L * G
    j = tl.where(fila_ok, filas // G, 0)
    g = filas % G
    pos = seq + j
    d = tl.arange(0, 64)

    base_q = QKV + (b * L + j).to(tl.int64) * paso_qkv + (h * G + g) * 64
    q = _norma_rope(base_q, pos, ROPE, qk_escala, CON_ROPE).to(tl.bfloat16)  # filas de relleno: copias de la fila 0

    # Las llaves ya guardadas que ve alguna fila del bloque: [lo, seq). La ventana de la fila j empieza en seq + j - V + 1.
    if VENTANA > 0:
        lo = tl.maximum(seq - VENTANA + 1, 0)
    else:
        lo = seq * 0
    inicio = lo + s * chunk
    fin = tl.minimum(inicio + chunk, seq)
    escala = tl.load(KS) * escala_log2

    m = tl.full([BLOQUE_M], VACIO, tl.float32)
    suma = tl.zeros([BLOQUE_M], tl.float32)
    acc = tl.zeros([BLOQUE_M, 64], tl.float32)
    # La tabla de paginas de la vuelta siguiente se pide antes de leer las llaves de esta: asi la lectura de la tabla
    # y la de las filas sueltas del cache que dependen de ella no se esperan una a la otra.
    fila_sig = _filas(TABLA, b, paso_tabla, inicio + tl.arange(0, BLOQUE_N), fin, PAGINA)
    for t0 in range(inicio, fin, BLOQUE_N):
        t = t0 + tl.arange(0, BLOQUE_N)
        ok = t < fin
        fila_c = fila_sig
        fila_sig = _filas(TABLA, b, paso_tabla, t + BLOQUE_N, fin, PAGINA)
        off = fila_c[:, None] * (HKV * 64) + h * 64 + d[None, :]
        k = tl.load(KC + off, mask=ok[:, None], other=0.0).to(tl.bfloat16)
        v = tl.load(VC + off, mask=ok[:, None], other=0.0).to(tl.bfloat16)
        if EN_CPU:  # el interprete de Triton multiplica mal en bf16: alli, en float32 exacto
            sc = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32)), input_precision="ieee") * escala
        else:
            sc = tl.dot(q, tl.trans(k)) * escala
        valido = ok[None, :]
        if VENTANA > 0:
            valido = valido & (t[None, :] > pos[:, None] - VENTANA)
        sc = tl.where(valido, sc, float("-inf"))
        m_nuevo = tl.maximum(m, tl.max(sc, 1))
        alfa = tl.math.exp2(m - m_nuevo)
        p = tl.math.exp2(sc - m_nuevo[:, None])
        suma = suma * alfa + tl.sum(p, 1)
        if EN_CPU:
            acc = acc * alfa[:, None] + tl.dot(p.to(tl.bfloat16).to(tl.float32), v.to(tl.float32),
                                               input_precision="ieee")
        else:
            acc = acc * alfa[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m = m_nuevo

    donde = (bh * n_pedazos + s) * BLOQUE_M + filas
    tl.store(MP + donde, m, mask=fila_ok)
    tl.store(LP + donde, suma, mask=fila_ok)
    tl.store(AP + donde[:, None] * 64 + d[None, :], acc, mask=fila_ok[:, None])


@triton.jit
def _final_v2(QKV, SEQ, LOC, KC, VC, KS, VS, ROPE, MP, LP, AP, OUT, paso_qkv, n_pedazos, qk_escala, escala_log2,
           HQ: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr, G: tl.constexpr, CON_ROPE: tl.constexpr,
           BLOQUE_M: tl.constexpr, BLOQUE_S: tl.constexpr, BLOQUE_L: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // (L * G)
    r = pid % (L * G)
    b = bh // HKV
    h = bh % HKV
    j = r // G
    cabeza = h * G + r % G
    seq = tl.load(SEQ + b)
    d = tl.arange(0, 64)
    uno = tl.arange(0, 1)
    ks, vs = tl.load(KS), tl.load(VS)

    # Juntar los pedazos de _parcial de a BLOQUE_S (composicion asociativa del softmax en linea).
    m = tl.max(tl.full([BLOQUE_S], VACIO, tl.float32), 0)
    suma = tl.sum(tl.zeros([BLOQUE_S], tl.float32), 0)
    acc = tl.zeros([64], tl.float32)
    for s0 in range(0, n_pedazos, BLOQUE_S):
        sidx = s0 + tl.arange(0, BLOQUE_S)
        hay = sidx < n_pedazos
        donde = (bh * n_pedazos + sidx) * BLOQUE_M + r
        ms = tl.load(MP + donde, mask=hay, other=VACIO)
        m_nuevo = tl.maximum(m, tl.max(ms, 0))
        alfa = tl.math.exp2(m - m_nuevo)
        w = tl.where(hay, tl.math.exp2(ms - m_nuevo), 0.0)
        suma = suma * alfa + tl.sum(tl.load(LP + donde, mask=hay, other=0.0) * w, 0)
        acc = acc * alfa + tl.sum(tl.load(AP + donde[:, None] * 64 + d[None, :], mask=hay[:, None], other=0.0)
                                  * w[:, None], 0)
        m = m_nuevo

    fila_q = QKV + (b * L + j).to(tl.int64) * paso_qkv + cabeza * 64 + uno
    q = tl.reshape(_norma_rope(fila_q, seq + j + uno, ROPE, qk_escala, CON_ROPE), (64,))
    q = q.to(tl.bfloat16).to(tl.float32)
    escribe = r == 0  # una fila por (peticion, cabeza kv) guarda las llaves y valores nuevos

    # Las L piezas del bloque, todas a la vez: cada fila j ve las i <= j (siempre dentro de la ventana).
    li = tl.arange(0, BLOQUE_L)
    hay_l = li < L
    i_c = tl.where(hay_l, li, 0)
    tok = (b * L + i_c).to(tl.int64)
    k8 = _a_fp8(_norma_rope(QKV + tok * paso_qkv + (HQ + h) * 64, seq + i_c, ROPE, 1.0, CON_ROPE) / ks)
    v8 = _a_fp8(tl.load(QKV + tok[:, None] * paso_qkv + (HQ + HKV + h) * 64 + d[None, :]).to(tl.float32) / vs)
    destino = tl.load(LOC + tok)[:, None] * (HKV * 64) + h * 64 + d[None, :]
    tl.store(KC + destino, k8, mask=escribe & hay_l[:, None])
    tl.store(VC + destino, v8, mask=escribe & hay_l[:, None])
    sc = tl.sum(q[None, :] * k8.to(tl.float32), 1) * ks * escala_log2
    sc = tl.where(hay_l & (li <= j), sc, float("-inf"))
    m_nuevo = tl.maximum(m, tl.max(sc, 0))
    alfa = tl.math.exp2(m - m_nuevo)
    p = tl.math.exp2(sc - m_nuevo)
    suma = suma * alfa + tl.sum(p, 0)
    acc = acc * alfa + tl.sum(p[:, None] * v8.to(tl.float32), 0)

    o = acc / suma * vs
    o = o * tl.rsqrt(tl.sum(o * o, 0) / 64 + 1e-5)
    puerta = tl.load(QKV + (b * L + j).to(tl.int64) * paso_qkv + (HQ + 2 * HKV + cabeza) * 64 + d).to(tl.float32)
    o = o * tl.sigmoid(puerta)
    tl.store(OUT + (b * L + j).to(tl.int64) * (HQ * 64) + cabeza * 64 + d, o.to(tl.bfloat16))


def _metodo_quorum(data, config):
    cfg = data["config"]
    batch, largo = cfg["batch"], cfg["q_len"]
    hq, hkv, dim, pagina = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    assert dim == 64 and hq % hkv == 0
    g = hq // hkv
    ventana = cfg["window"]
    qkv = data["qkv"].reshape(batch * largo, -1)
    tabla = data["page_table"]
    device = qkv.device
    con_rope = bool(cfg["rope_theta"])
    rope = _cos_sin(cfg["rope_theta"], dim, device) if con_rope else qkv  # sin RoPE no se lee
    qk_escala = float(cfg["qk_scale_factor"])
    escala_log2 = float(cfg["sm_scale"]) * LOG2E
    # Energia de referencia: el techo de Cauchy-Schwarz del logit en base 2, |q| |k| sm_scale log2(e) con
    # |q| = qk_scale_factor * sqrt(64) y |k| = sqrt(64) (QK-norm), menos la holgura.
    c_ref = qk_escala * dim * escala_log2 - HOLGURA

    bloque_n, etapas, warps, por_sm, grupos, prefetch, qk_fp8, tandas, min_chunk = config
    filas_cabeza = largo * g
    while grupos > 1 and filas_cabeza % grupos:
        grupos //= 2
    filas = max(16, triton.next_power_of_2(filas_cabeza // grupos))
    n_trabajos = max(1, tandas * _n_sms(device) * por_sm // (hkv * grupos))
    n_grupos = batch * hkv * grupos
    acc, suma, cnt = _estado(device, n_grupos * filas, n_grupos)
    out = torch.empty((batch, largo, hq * 64), dtype=torch.bfloat16, device=device)
    en_cpu = device.type == "cpu"

    _atencion[(grupos * hkv, n_trabajos + batch)](
        qkv, data["seq_lens"], data["cache_loc"], tabla, data["k_cache"], data["v_cache"], data["k_scale"],
        data["v_scale"], rope, acc, suma, cnt, out, qkv.stride(0), tabla.stride(0), n_trabajos, min_chunk,
        qk_escala, escala_log2, c_ref,
        B=batch, BB=triton.next_power_of_2(batch), HQ=hq, HKV=hkv, L=largo, G=g, R=grupos, FILAS=filas,
        PAGINA=pagina, VENTANA=ventana, CON_ROPE=con_rope, BLOQUE_N=bloque_n, ETAPAS=etapas,
        BLOQUE_L=max(16, triton.next_power_of_2(largo)), PREFETCH=prefetch, QK_FP8=qk_fp8,
        QUORUM=QUORUM, EN_CPU=en_cpu, num_warps=warps)
    if not QUORUM:
        _cierre[(n_grupos,)](qkv, acc, suma, data["v_scale"], out, qkv.stride(0), HQ=hq, HKV=hkv, L=largo, G=g,
                             R=grupos, FILAS=filas, num_warps=4)
    return out


# --- v11: el pedazo de decode con contexto completo en CUDA nativo ------------------------------------------------------
# Un bloque de 128 hilos por (pedazo, peticion) atiende las dos cabezas kv: lee cada fila de 128 bytes del cache entera,
# en un anillo de 4 etapas de cp.async (con swizzle contra conflictos de bancos), y multiplica en tensor cores
# (mma.sync m16n8k16) con la energia de referencia fija de la v6, que evita reescalar los acumuladores. Escribe los
# pedazos en el formato de la v2, asi que _final_v2 los junta igual. Se compila con load_inline la primera vez; si algo
# falla, imprime el error y queda el camino de Triton.
_CUDA_FUENTE = r"""// Atencion de un paso de decodificacion sobre el contexto completo (Muse Spark Open, TP8: 16 cabezas de consulta,
// 2 cabezas kv de 64, cache FP8 E4M3 en filas sueltas): los pedazos del contexto ya guardado, en el formato de la v2
// (m, l, acumulado por pedazo y fila), para que _final_v2 los junte con las piezas nuevas.
//
// Un bloque de 128 hilos por (pedazo, peticion) atiende las DOS cabezas kv: cada fila del cache (128 bytes: k de las
// dos cabezas) se lee entera, en copias asincronas de 16 bytes (cp.async) a un anillo de ETAPAS bloques de llaves en
// memoria compartida. Producto q.k y suma p.v en tensor cores (mma.sync m16n8k16): las llaves E4M3 pasan a f16 en
// registros (exacto: E4M3 cabe en f16) y los valores a bf16 (como p, que necesita el rango de bf16).
//
// Softmax con energia de referencia fija (la de la v6): con QK-norm, Cauchy-Schwarz acota cada logit, y con c = techo
// - 40 todos los terminos 2^(s - c) caen en [2^-86, 2^40]. No hay maximo ni reescalado: los acumuladores de los tensor
// cores nunca se tocan, y el pedazo sale con m = c.

#ifdef __CUDACC_RTC__  // NVRTC (la revision de compilacion sin GPU) no trae la biblioteca de C
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
#else
#include <stdint.h>
#endif
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace muse {

constexpr int HILOS = 128;
constexpr int LLAVES = 64;           // llaves por vuelta (4 warps x 16)
constexpr int FILA = 128;            // bytes por fila del cache: 2 cabezas kv x 64
constexpr int ETAPAS = 4;            // bloques de llaves en el anillo
constexpr int BLOQUE_ETAPA = 2 * LLAVES * FILA;  // k y v de una vuelta: 16 KB

__device__ __forceinline__ uint32_t a_smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t destino, const void* origen) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(destino), "l"(origen));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// Swizzle: el trozo de 16 bytes c de la fila f se guarda en el lugar c ^ (f & 7), para que las lecturas de 8 filas
// seguidas en la misma columna caigan en bancos distintos.
__device__ __forceinline__ int lugar(int fila, int byte) {
    return fila * FILA + ((((byte >> 4) ^ (fila & 7)) << 4) | (byte & 15));
}

__device__ __forceinline__ uint32_t e4m3x2_a_f16x2(uint16_t dos) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(dos), __NV_E4M3);
    return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}

__device__ __forceinline__ uint32_t e4m3x2_a_bf16x2(uint16_t dos) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(dos), __NV_E4M3);
    __half2 hh = *reinterpret_cast<__half2*>(&h);
    float2 f = __half22float2(hh);
    __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
    return *reinterpret_cast<uint32_t*>(&b);
}

__device__ __forceinline__ uint32_t f32x2_a_bf16x2(float x, float y) {
    __nv_bfloat162 b = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&b);
}

__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
    // filas 8-15 de A en cero (relleno: solo hay 8 filas de consulta por cabeza kv)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float exp2_rapido(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

// Fila del cache de la posicion t de la peticion b.
__device__ __forceinline__ int64_t fila_cache(const int* tabla, int64_t paso_tabla, int b, int t, int pagina) {
    return static_cast<int64_t>(tabla[b * paso_tabla + t / pagina]) * pagina + t % pagina;
}

extern "C" __global__ void __launch_bounds__(HILOS)
parcial_completo(const __nv_bfloat16* __restrict__ qkv, int64_t paso_qkv, const int* __restrict__ seq_lens,
                 const int* __restrict__ tabla, int64_t paso_tabla, int pagina, const uint8_t* __restrict__ kc,
                 const uint8_t* __restrict__ vc, const float* __restrict__ k_scale, float qk_escala,
                 float escala_log2, float c_ref, float* __restrict__ mp, float* __restrict__ lp,
                 float* __restrict__ ap, int chunk, int n_pedazos, int bloque_m) {
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(16) __half q_s[16][64];  // las 16 cabezas de consulta, normalizadas y por qk_escala
    const int s = blockIdx.x, b = blockIdx.y;
    const int hilo = threadIdx.x, warp = hilo >> 5, carril = hilo & 31;
    const int g = carril >> 2, tig = carril & 3;   // grupo y posicion en el grupo (fragmentos de mma.sync)

    const int seq = seq_lens[b];
    const int inicio = s * chunk;
    const int fin = min(inicio + chunk, seq);
    const int n_vueltas = fin > inicio ? (fin - inicio + LLAVES - 1) / LLAVES : 0;

    // Que copia este hilo: la fila hilo / 2 de la vuelta, mitad (hilo & 1) de los 128 bytes, de k y de v.
    const int fila_c = hilo >> 1, mitad = hilo & 1;
    auto pedir = [&](int vuelta, int64_t fila) {
        uint8_t* base = smem + (vuelta % ETAPAS) * BLOQUE_ETAPA;
        const uint8_t* k = kc + fila * FILA + mitad * 64;
        const uint8_t* v = vc + fila * FILA + mitad * 64;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int byte = mitad * 64 + c * 16;
            cp_async16(a_smem(base + lugar(fila_c, byte)), k + c * 16);
            cp_async16(a_smem(base + LLAVES * FILA + lugar(fila_c, byte)), v + c * 16);
        }
    };
    auto fila_de = [&](int vuelta) -> int64_t {
        const int t = inicio + vuelta * LLAVES + fila_c;
        return t < fin ? fila_cache(tabla, paso_tabla, b, t, pagina) : 0;  // fuera del pedazo: la fila 0 (se enmascara)
    };

    // Prologo: las primeras ETAPAS - 1 vueltas en camino mientras se normalizan las consultas.
    int64_t fila_sig = 0;
#pragma unroll
    for (int v = 0; v < ETAPAS - 1; ++v) {
        if (v < n_vueltas) pedir(v, fila_de(v));
        cp_async_commit();
    }
    if (ETAPAS - 1 < n_vueltas) fila_sig = fila_de(ETAPAS - 1);

    {   // RMSNorm de las 16 cabezas de consulta (8 hilos por cabeza, 8 valores cada uno), por qk_escala, en f16.
        const int cabeza = hilo >> 3, parte = (hilo & 7) * 8;
        const __nv_bfloat16* fila_q = qkv + b * paso_qkv + cabeza * 64 + parte;
        float x[8], ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) { x[i] = __bfloat162float(fila_q[i]); ss += x[i] * x[i]; }
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float inv = rsqrtf(ss / 64.f + 1e-5f) * qk_escala;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            // como el resto del kernel: la consulta normalizada redondeada a bf16 (la referencia la redondea a bf16)
            q_s[cabeza][parte + i] = __float2half(__bfloat162float(__float2bfloat16(x[i] * inv)));
        }
    }
    __syncthreads();

    // Fragmentos A de q (constantes): cabeza kv h, fila g = cabeza de consulta h * 8 + g, 4 pasos de 16 dimensiones.
    uint32_t qa[2][4][2];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            qa[h][kk][0] = *reinterpret_cast<const uint32_t*>(&q_s[h * 8 + g][kk * 16 + tig * 2]);
            qa[h][kk][1] = *reinterpret_cast<const uint32_t*>(&q_s[h * 8 + g][kk * 16 + 8 + tig * 2]);
        }
    const float escala = k_scale[0] * escala_log2;

    float o[2][8][4];   // acumulado p.v: cabeza kv, bloque de 8 dimensiones, fragmento C (filas g y g+8)
    float l[2] = {0.f, 0.f};
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int nd = 0; nd < 8; ++nd)
#pragma unroll
            for (int i = 0; i < 4; ++i) o[h][nd][i] = 0.f;

    for (int vuelta = 0; vuelta < n_vueltas; ++vuelta) {
        cp_async_wait<ETAPAS - 2>();
        __syncthreads();   // la vuelta llego para todos, y la ranura de la vuelta anterior ya nadie la lee
        if (vuelta + ETAPAS - 1 < n_vueltas) pedir(vuelta + ETAPAS - 1, fila_sig);
        cp_async_commit();
        if (vuelta + ETAPAS < n_vueltas) fila_sig = fila_de(vuelta + ETAPAS);

        const uint8_t* ks = smem + (vuelta % ETAPAS) * BLOQUE_ETAPA;
        const uint8_t* vs = ks + LLAVES * FILA;
        const int llave0 = warp * 16;                 // las 16 llaves de este warp en la vuelta
        const int t0 = inicio + vuelta * LLAVES + llave0;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            // s = q . k: 2 bloques de 8 llaves x 4 pasos de 16 dimensiones.
            float sc[2][4];
#pragma unroll
            for (int nb = 0; nb < 2; ++nb) {
                sc[nb][0] = sc[nb][1] = sc[nb][2] = sc[nb][3] = 0.f;
                const int fila = llave0 + nb * 8 + g;
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int byte = h * 64 + kk * 16 + tig * 2;
                    const uint32_t b0 = e4m3x2_a_f16x2(*reinterpret_cast<const uint16_t*>(ks + lugar(fila, byte)));
                    const uint32_t b1 = e4m3x2_a_f16x2(*reinterpret_cast<const uint16_t*>(ks + lugar(fila, byte + 8)));
                    mma_f16(sc[nb], qa[h][kk][0], qa[h][kk][1], b0, b1);
                }
            }
            // p = 2^(s - c) (fila g; las llaves tig*2 y tig*2+1 de cada bloque de 8), 0 fuera del pedazo.
            float p[2][2];
#pragma unroll
            for (int nb = 0; nb < 2; ++nb)
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int t = t0 + nb * 8 + tig * 2 + i;
                    const float e = fminf(sc[nb][i] * escala - c_ref, 100.f);
                    p[nb][i] = t < fin ? exp2_rapido(e) : 0.f;
                    l[h] += p[nb][i];
                }
            // o += p . v: A = p (16 llaves de este warp), B = v (las mismas llaves, 8 dimensiones por bloque).
            const uint32_t pa0 = f32x2_a_bf16x2(p[0][0], p[0][1]);
            const uint32_t pa2 = f32x2_a_bf16x2(p[1][0], p[1][1]);
            const int fv0 = llave0 + tig * 2, fv8 = llave0 + 8 + tig * 2;
#pragma unroll
            for (int nd = 0; nd < 8; ++nd) {
                const int byte = h * 64 + nd * 8 + g;
                const uint16_t v0 = vs[lugar(fv0, byte)] | (static_cast<uint16_t>(vs[lugar(fv0 + 1, byte)]) << 8);
                const uint16_t v8 = vs[lugar(fv8, byte)] | (static_cast<uint16_t>(vs[lugar(fv8 + 1, byte)]) << 8);
                mma_bf16(o[h][nd], pa0, pa2, e4m3x2_a_bf16x2(v0), e4m3x2_a_bf16x2(v8));
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    // Juntar los 4 warps (cada uno vio 16 de cada 64 llaves) en la memoria compartida, ya libre.
    float* red_o = reinterpret_cast<float*>(smem);            // [4 warps][2 cabezas][8 filas][64]
    float* red_l = red_o + 4 * 2 * 8 * 64;                     // [4 warps][2 cabezas][8 filas]
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        float lh = l[h];
        lh += __shfl_xor_sync(0xffffffffu, lh, 1);
        lh += __shfl_xor_sync(0xffffffffu, lh, 2);
        if (tig == 0) red_l[(warp * 2 + h) * 8 + g] = lh;
#pragma unroll
        for (int nd = 0; nd < 8; ++nd) {
            float* d = red_o + ((warp * 2 + h) * 8 + g) * 64 + nd * 8 + tig * 2;
            d[0] = o[h][nd][0];
            d[1] = o[h][nd][1];
        }
    }
    __syncthreads();
    for (int i = hilo; i < 2 * 8 * 64; i += HILOS) {
        const int h = i / 512, fila = (i / 64) % 8, dim = i % 64;
        float acc = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w) acc += red_o[((w * 2 + h) * 8 + fila) * 64 + dim];
        const int64_t donde = (static_cast<int64_t>(b * 2 + h) * n_pedazos + s) * bloque_m + fila;
        ap[donde * 64 + dim] = acc;
        if (dim == 0) {
            float suma = 0.f;
#pragma unroll
            for (int w = 0; w < 4; ++w) suma += red_l[(w * 2 + h) * 8 + fila];
            lp[donde] = suma;
            mp[donde] = c_ref;
        }
    }
}

}  // namespace muse
"""

_CUDA_ANFITRION = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void parcial_completo_lanzar(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor tabla, torch::Tensor kc,
                             torch::Tensor vc, torch::Tensor k_scale, torch::Tensor mp, torch::Tensor lp,
                             torch::Tensor ap, int64_t pagina, int64_t chunk, int64_t n_pedazos, int64_t bloque_m,
                             double qk_escala, double escala_log2, double c_ref) {
    const int memoria = muse::ETAPAS * muse::BLOQUE_ETAPA;
    static bool listo = false;
    if (!listo) {
        cudaFuncSetAttribute(muse::parcial_completo, cudaFuncAttributeMaxDynamicSharedMemorySize, memoria);
        listo = true;
    }
    dim3 malla(static_cast<unsigned>(n_pedazos), static_cast<unsigned>(seq_lens.size(0)));
    muse::parcial_completo<<<malla, muse::HILOS, memoria, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), qkv.stride(0), seq_lens.data_ptr<int>(),
        tabla.data_ptr<int>(), tabla.stride(0), static_cast<int>(pagina),
        reinterpret_cast<const uint8_t*>(kc.data_ptr()), reinterpret_cast<const uint8_t*>(vc.data_ptr()),
        k_scale.data_ptr<float>(), static_cast<float>(qk_escala), static_cast<float>(escala_log2),
        static_cast<float>(c_ref), mp.data_ptr<float>(), lp.data_ptr<float>(), ap.data_ptr<float>(),
        static_cast<int>(chunk), static_cast<int>(n_pedazos), static_cast<int>(bloque_m));
}
"""

_CUDA = {}


def _modulo_cuda():
    """El modulo compilado (una vez), o None si no se pudo: entonces se usa el camino de Triton."""
    if "modulo" not in _CUDA:
        try:
            from torch.utils.cpp_extension import load_inline
            _CUDA["modulo"] = load_inline(
                name="muse_parcial_completo_v11", cpp_sources="""
void parcial_completo_lanzar(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor tabla, torch::Tensor kc,
                             torch::Tensor vc, torch::Tensor k_scale, torch::Tensor mp, torch::Tensor lp,
                             torch::Tensor ap, int64_t pagina, int64_t chunk, int64_t n_pedazos, int64_t bloque_m,
                             double qk_escala, double escala_log2, double c_ref);""",
                cuda_sources=_CUDA_FUENTE + _CUDA_ANFITRION, functions=["parcial_completo_lanzar"],
                extra_cuda_cflags=["-O3", "-std=c++17"], verbose=False)
            print("v11: kernel CUDA compilado", flush=True)
        except Exception as error:  # el juez no se cae: queda Triton, y el log dice por que
            print(f"v11: el kernel CUDA no compilo, sigo con Triton: {type(error).__name__}: {error}"[:4000], flush=True)
            _CUDA["modulo"] = None
    return _CUDA["modulo"]


def _metodo_pedazos(data):
    cfg = data["config"]
    batch, largo = cfg["batch"], cfg["q_len"]
    hq, hkv, dim, pagina = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    assert dim == 64 and hq % hkv == 0
    g = hq // hkv
    ventana = cfg["window"]
    qkv = data["qkv"].reshape(batch * largo, -1)
    tabla = data["page_table"]
    device = qkv.device
    con_rope = bool(cfg["rope_theta"])
    rope = _cos_sin(cfg["rope_theta"], dim, device) if con_rope else qkv  # sin RoPE no se lee
    qk_escala = float(cfg["qk_scale_factor"])
    escala_log2 = float(cfg["sm_scale"]) * LOG2E

    # Pedazos del contexto ya guardado, por la forma (no por los valores): unos PROGRAMAS_POR_SM programas por SM,
    # de al menos dos vueltas de bloque_n llaves cada uno.
    max_llaves = ventana - 1 if ventana else tabla.shape[1] * pagina
    bloque_n = 64 if max_llaves <= 4096 else 128
    n_bh = batch * hkv
    n_pedazos = max(1, min(triton.cdiv(PROGRAMAS_POR_SM * _n_sms(device), n_bh),
                           triton.cdiv(max_llaves, 2 * bloque_n)))
    chunk = triton.cdiv(triton.cdiv(max_llaves, n_pedazos), bloque_n) * bloque_n
    n_pedazos = max(1, triton.cdiv(max_llaves, chunk))
    bloque_m = max(16, triton.next_power_of_2(largo * g))

    mp = torch.empty((n_bh * n_pedazos * bloque_m,), dtype=torch.float32, device=device)
    lp = torch.empty_like(mp)
    ap = torch.empty((n_bh * n_pedazos * bloque_m, 64), dtype=torch.float32, device=device)
    out = torch.empty((batch, largo, hq * 64), dtype=torch.bfloat16, device=device)
    k_cache, v_cache = data["k_cache"], data["v_cache"]

    modulo = _modulo_cuda() if (not ventana and largo == 1 and hkv == 2 and device.type == "cuda") else None
    if modulo is not None:
        c_ref = qk_escala * dim * escala_log2 - HOLGURA  # la energia de referencia de la v6 (ver _metodo_quorum)
        modulo.parcial_completo_lanzar(qkv, data["seq_lens"], tabla, k_cache, v_cache, data["k_scale"], mp, lp, ap,
                                       pagina, chunk, n_pedazos, bloque_m, qk_escala, escala_log2, c_ref)
    else:
        _parcial_v2[(n_pedazos, n_bh)](
            qkv, data["seq_lens"], tabla, k_cache, v_cache, data["k_scale"], rope, mp, lp, ap,
            qkv.stride(0), tabla.stride(0), chunk, n_pedazos, qk_escala, escala_log2,
            HQ=hq, HKV=hkv, L=largo, G=g, PAGINA=pagina, VENTANA=ventana, CON_ROPE=con_rope, BLOQUE_M=bloque_m,
            BLOQUE_N=bloque_n, EN_CPU=device.type == "cpu", num_warps=4, num_stages=2)
    _final_v2[(n_bh * largo * g,)](
        qkv, data["seq_lens"], data["cache_loc"], k_cache, v_cache, data["k_scale"], data["v_scale"], rope,
        mp, lp, ap, out, qkv.stride(0), n_pedazos, qk_escala, escala_log2,
        HQ=hq, HKV=hkv, L=largo, G=g, CON_ROPE=con_rope, BLOQUE_M=bloque_m,
        BLOQUE_S=min(BLOQUE_S_V2, triton.next_power_of_2(n_pedazos)), BLOQUE_L=triton.next_power_of_2(largo),
        num_warps=4)
    return out


# --- v8: cada forma con el metodo que gano en el juez ------------------------------------------------------------------
# Medido en la H200 (v2 = pedazos, v6/v7 = quorum, en us por llamada):
#   pocas peticiones, contexto completo: quorum (decode 1: 17 vs 24; verify 1: 30 vs 35; verify 16: 243 vs 269)
#   decode con ventana, 1 peticion: quorum (6.0 vs 6.9); verify con ventana y muchas peticiones: quorum con
#   trabajos grandes y consultas FP8 (verify 32: 21.9 vs 38.6)
#   el resto (16-32 peticiones en decode, verify con ventana y pocas peticiones, verify 32 completo): pedazos
# Las reglas dependen solo de la forma (q_len, batch, tipo de capa) y cambian por tramos de batch, no por un valor
# exacto: las formas vecinas usan el mismo metodo.
QUORUM_VENTANA_VERIFY = (64, 3, 4, 3, 1, 0, True, 1, 512)  # la configuracion de la v6 para verify con ventana
# v10: decode con ventana y pocas peticiones con la configuracion de la v6 (6.0 us en la H200; la de la v7, 7.0-7.2).
# Probado en la v9 junto con dos experimentos que se descartaron: ranuras en verify completo (el que cierra suma cientos
# de ranuras en serie: 190 us en verify 1, contra 30 con atomicos) y 3 etapas en decode completo (194/361 contra 165/310).
QUORUM_VENTANA_DECODE = (64, 2, 4, 4, 1, 0, False, 2, 64)


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, largo, completo = cfg["batch"], cfg["q_len"], not cfg["window"]
    l = 1 if largo == 1 else 8
    if completo:
        if batch <= 4 or (l == 8 and batch <= 16):
            return _metodo_quorum(data, CONFIG[(l, True)])
        return _metodo_pedazos(data)
    if l == 1:
        return _metodo_quorum(data, QUORUM_VENTANA_DECODE) if batch <= 4 else _metodo_pedazos(data)
    return _metodo_quorum(data, QUORUM_VENTANA_VERIFY) if batch >= 24 else _metodo_pedazos(data)
