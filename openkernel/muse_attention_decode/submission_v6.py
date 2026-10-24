# okc: problem=muse_attention_decode
"""Atencion con puerta de Muse Spark Open (un paso de decodificacion o un bloque de verificacion de 8 piezas) en UN
solo kernel de Triton, en vez de los cinco de SGLang (FusedQKNorm, store_cache, q a FP8, FA3 y rms_norm_gate).

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
CONFIG = {(1, True): (128, 3, 4, 4, 1, 2, False, 2, 512), (1, False): (64, 2, 4, 4, 1, 0, False, 2, 64),
          (8, True): (64, 4, 4, 4, 1, 2, True, 1, 1024), (8, False): (64, 3, 4, 3, 1, 0, True, 1, 512)}
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
        k = tl.load(KC + off, mask=ok[:, None], other=0.0, eviction_policy="evict_first")
        v = tl.load(VC + off, mask=ok[:, None], other=0.0, eviction_policy="evict_first")
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


def kernel_fn(data: Inputs) -> torch.Tensor:
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

    bloque_n, etapas, warps, por_sm, grupos, prefetch, qk_fp8, tandas, min_chunk = \
        CONFIG[(1 if largo == 1 else 8, not ventana)]
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
