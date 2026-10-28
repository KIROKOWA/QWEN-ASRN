# okc: problem=muse_attention_decode
"""MuseOne CUDA — candidato experimental derivado de KIROKOWA v17.

Ruta nueva: verify/full con batch >= 16, CUDA native warp-local + cierre CUDA.
Decode, sliding-window y verify/full pequenos conservan los kernels de v17.

AVISO: verificaciones estaticas superadas; CUDA/PTX y 16 casos del juez PENDIENTES.

Base original: candidato hibrido basado en el perfil H200 de v16.

Verify con contexto completo usa FlashAttention-3 de la biblioteca instalada,
con QK-norm/escritura de cache/preparacion FP8 fusionadas y RMSNorm+gate propio.
Las demas rutas conservan v16. Si no existe una API FA3 compatible, se usa v16.
La seleccion depende solo de config, dispositivo y API publica de la biblioteca.
No importa baseline/reference/harness, no omite claves y no conserva respuestas.

Antecedente comunicado: v16 paso 16 pruebas y obtuvo 1.348495x en H200; el
kernel verify/full consumio 267.1/518.1 us en lotes 16/32 del perfil enviado.
Esta v17 no se ha medido en GPU. No implica validacion oficial ni ganancia.
"""

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
    # El quorum ya esta completo. Intercambiar por cero lee y limpia sin
    # una segunda escritura global; no se reutiliza ningun resultado anterior.
    p_acc = ACC + donde[:, None] * 64 + d[None, :]
    acc = tl.atomic_xchg(p_acc, tl.zeros([FILAS, 64], tl.float32), mask=fila_ok[:, None], sem="relaxed")
    suma = tl.atomic_xchg(SUMA + donde, tl.zeros([FILAS], tl.float32), mask=fila_ok, sem="relaxed")
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
# v14: q.k en FP8 directo (las llaves no se convierten), v en bf16 una vez por vuelta y fragmentos con
# ldmatrix.trans; 71 registros, 64 KB de memoria compartida, 3 bloques por SM. v13: un bloque de 256 hilos (8 warps: un warp por cabeza kv y grupo de 16 llaves) por (pedazo, peticion), 80 registros
# por hilo y 3 bloques por SM (la v12: 219 registros, 2 bloques de 4 warps). Lee cada fila de 128 bytes del cache entera,
# en un anillo de 4 etapas de cp.async (con swizzle contra conflictos de bancos), y multiplica en tensor cores
# (mma.sync m16n8k16) con la energia de referencia fija de la v6, que evita reescalar los acumuladores. Escribe los
# pedazos en el formato de la v2, asi que _final_v2 los junta igual. Se compila con load_inline la primera vez; si algo
# falla, imprime el error y queda el camino de Triton.
_CUDA_FUENTE = r"""// Atencion de un paso de decodificacion sobre el contexto completo (Muse Spark Open, TP8: 16 cabezas de consulta,
// 2 cabezas kv de 64, cache FP8 E4M3 en filas sueltas): los pedazos del contexto ya guardado, en el formato de la v2
// (m, l, acumulado por pedazo y fila), para que _final_v2 los junte con las piezas nuevas.
//
// Un bloque de 256 hilos por (pedazo, peticion) atiende las DOS cabezas kv (un warp, una cabeza): cada fila del cache (128 bytes: k de las
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

constexpr int HILOS = 256;            // 8 warps: 2 cabezas kv x 4 grupos de 16 llaves
constexpr int BLOQUES_POR_SM = 3;   // __launch_bounds__: a lo sumo 65536 / (256 x 3) = 85 registros por hilo
constexpr int LLAVES = 64;           // llaves por vuelta (4 grupos de 16)
constexpr int FILA = 128;            // bytes por fila del cache: 2 cabezas kv x 64
constexpr int ETAPAS = 3;            // bloques de llaves en el anillo (v14: 3, para que quepan 3 bloques por SM)
constexpr int BLOQUE_ETAPA = 2 * LLAVES * FILA;  // k y v de una vuelta: 16 KB
constexpr int FILA_VB = 2 * FILA;   // v de una vuelta en bf16: 256 bytes por llave (16 trozos de 16 bytes)

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

// q.k en FP8 directo: A = q (E4M3, 16 x 32), B = k (E4M3, 32 x 8); filas 8-15 de A en cero (relleno).
__device__ __forceinline__ void mma_e4m3(float* c, uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

// Igual, con las 16 filas de A llenas (verify: 16 consultas reales por warp).
__device__ __forceinline__ void mma_e4m3_4(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                           uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Cuatro matrices 8 x 8 de 16 bits, traspuestas, de una vez: el fragmento A de v^T para mma.sync.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t* r, uint32_t direccion) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(direccion));
}

// Lugar del trozo de 16 bytes c (0..15) de la llave f en el bloque de v en bf16, con swizzle (contra conflictos de
// bancos de ldmatrix, que lee 8 llaves seguidas en la misma columna).
__device__ __forceinline__ int lugar_vb(int f, int c) { return f * FILA_VB + ((c ^ (f & 7)) << 4); }

__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
    // filas 8-15 de A en cero (relleno: solo hay 8 filas de consulta por cabeza kv)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Dos bytes E4M3 de la columna byte, de las filas f y f + 1, como bf16x2 (el de la fila f en la mitad baja).
__device__ __forceinline__ uint32_t v_par(const uint8_t* vs, int f, int byte) {
    return e4m3x2_a_bf16x2(vs[lugar(f, byte)] | (static_cast<uint16_t>(vs[lugar(f + 1, byte)]) << 8));
}

// Un valor que el compilador no puede sacar del bucle: asi las direcciones de memoria compartida que dependen de el se
// calculan en cada vuelta (unas pocas instrucciones enteras) en vez de guardarse ~48 precalculadas en registros.
__device__ __forceinline__ int opaco(int x) {
    int y;
    asm volatile("mov.b32 %0, %1;\n" : "=r"(y) : "r"(x));
    return y;
}

__device__ __forceinline__ float exp2_rapido(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

// Fila del cache de la posicion t de la peticion b (cabe en 32 bits: el cache tiene unos pocos millones de filas).
template <int PAGINA>
__device__ __forceinline__ int fila_cache(const int* tabla, int64_t paso_tabla, int b, int t, int pagina) {
    if constexpr (PAGINA == 1) {
        return tabla[b * paso_tabla + t];
    } else {
        return tabla[b * paso_tabla + t / PAGINA] * PAGINA + t % PAGINA;
    }
}

template <int PAGINA>
__global__ void __launch_bounds__(HILOS, BLOQUES_POR_SM)
parcial_completo(const __nv_bfloat16* __restrict__ qkv, int64_t paso_qkv, const int* __restrict__ seq_lens,
                 const int* __restrict__ tabla, int64_t paso_tabla, int pagina, const uint8_t* __restrict__ kc,
                 const uint8_t* __restrict__ vc, const float* __restrict__ k_scale, float qk_escala,
                 float escala_log2, float c_ref, float* __restrict__ lp,
                 float* __restrict__ ap, int chunk, int n_pedazos, int bloque_m) {
    extern __shared__ __align__(128) uint8_t smem[];        // anillo [ETAPAS][k | v] y despues v en bf16
    __shared__ __align__(16) uint8_t q_s[16][64];           // las 16 cabezas de consulta normalizadas, en E4M3
    uint8_t* vb = smem + ETAPAS * BLOQUE_ETAPA;              // v de la vuelta en bf16 [64 llaves][256 bytes]
    const int s = blockIdx.x, b = blockIdx.y;
    const int hilo = threadIdx.x, warp = hilo >> 5, carril = hilo & 31;
    const int g = carril >> 2, tig = carril & 3;   // grupo y posicion en el grupo (fragmentos de mma.sync)
    // Cada warp atiende UNA cabeza kv (h) y 16 de las 64 llaves de la vuelta (kg).
    const int h = warp >> 2, kg = warp & 3;

    const int seq = seq_lens[b];
    const int inicio = s * chunk;
    const int fin = min(inicio + chunk, seq);
    const int n_vueltas = fin > inicio ? (fin - inicio + LLAVES - 1) / LLAVES : 0;

    // Que copia (y despues convierte) este hilo: la fila hilo / 4 de la vuelta, cuarto (hilo & 3) de los 128 bytes.
    const int fila_c = hilo >> 2, cuarto = hilo & 3;
    auto pedir = [&](int vuelta, int fila32) {
        const int64_t fila = fila32;
        uint8_t* base = smem + (vuelta % ETAPAS) * BLOQUE_ETAPA;
        const uint8_t* k = kc + fila * FILA + cuarto * 32;
        const uint8_t* v = vc + fila * FILA + cuarto * 32;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int byte = cuarto * 32 + c * 16;
            cp_async16(a_smem(base + lugar(fila_c, byte)), k + c * 16);
            cp_async16(a_smem(base + LLAVES * FILA + lugar(fila_c, byte)), v + c * 16);
        }
    };
    auto fila_de = [&](int vuelta) -> int {
        const int t = inicio + vuelta * LLAVES + fila_c;
        return t < fin ? fila_cache<PAGINA>(tabla, paso_tabla, b, t, pagina) : 0;  // fuera del pedazo: la fila 0 (se enmascara)
    };

    int fila_sig = 0;
#pragma unroll
    for (int v = 0; v < ETAPAS - 1; ++v) {
        if (v < n_vueltas) pedir(v, fila_de(v));
        cp_async_commit();
    }
    if (ETAPAS - 1 < n_vueltas) fila_sig = fila_de(ETAPAS - 1);

    if (hilo < 128) {   // RMSNorm de las 16 cabezas de consulta, en E4M3 (como FA3; qk_escala va en la escala)
        const int cabeza = hilo >> 3, parte = (hilo & 7) * 8;
        const __nv_bfloat16* fila_q = qkv + b * paso_qkv + cabeza * 64 + parte;
        float x[8], ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) { x[i] = __bfloat162float(fila_q[i]); ss += x[i] * x[i]; }
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float inv = rsqrtf(ss / 64.f + 1e-5f);
#pragma unroll
        for (int i = 0; i < 8; ++i)
            q_s[cabeza][parte + i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(x[i] * inv, __NV_SATFINITE, __NV_E4M3));
    }
    __syncthreads();
    const float escala = k_scale[0] * escala_log2 * qk_escala;

    // Acumulado traspuesto, o^T = v^T p^T (16 dimensiones por bloque, sin filas de relleno).
    float o[4][4];
    float l = 0.f;
#pragma unroll
    for (int md = 0; md < 4; ++md)
#pragma unroll
        for (int i = 0; i < 4; ++i) o[md][i] = 0.f;

    const int llave0 = kg * 16;
#pragma unroll 1
    for (int vuelta = 0; vuelta < n_vueltas; ++vuelta) {
        cp_async_wait<ETAPAS - 2>();
        __syncthreads();   // la vuelta llego; nadie lee ya la ranura de la vuelta anterior ni vb
        if (vuelta + ETAPAS - 1 < n_vueltas) pedir(vuelta + ETAPAS - 1, fila_sig);
        cp_async_commit();
        if (vuelta + ETAPAS < n_vueltas) fila_sig = fila_de(vuelta + ETAPAS);

        const uint8_t* ks = smem + (vuelta % ETAPAS) * BLOQUE_ETAPA;
        const uint8_t* vs = ks + LLAVES * FILA;
        // v E4M3 -> bf16, una vez por vuelta y entre todos: 32 bytes por hilo (su fila y su cuarto).
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const uint4 crudo = *reinterpret_cast<const uint4*>(vs + lugar(fila_c, cuarto * 32 + c * 16));
            const uint32_t pal[4] = {crudo.x, crudo.y, crudo.z, crudo.w};
            uint32_t sal[8];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                sal[2 * w] = e4m3x2_a_bf16x2(static_cast<uint16_t>(pal[w] & 0xffffu));
                sal[2 * w + 1] = e4m3x2_a_bf16x2(static_cast<uint16_t>(pal[w] >> 16));
            }
            const int trozo = (cuarto * 32 + c * 16) / 8;  // 16 bytes E4M3 = 2 trozos de 16 bytes en bf16
            *reinterpret_cast<uint4*>(vb + lugar_vb(fila_c, trozo)) = make_uint4(sal[0], sal[1], sal[2], sal[3]);
            *reinterpret_cast<uint4*>(vb + lugar_vb(fila_c, trozo + 1)) = make_uint4(sal[4], sal[5], sal[6], sal[7]);
        }

        // s = q . k en FP8 (2 pasos de 32 dimensiones) para 2 bloques de 8 llaves; p = 2^(s - c).
        const int t0 = inicio + vuelta * LLAVES + llave0;
        uint32_t pa[2];
#pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
            float sc[4] = {0.f, 0.f, 0.f, 0.f};
            const int fila = llave0 + nb * 8 + g;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                const int byte = h * 64 + kk * 32 + tig * 4;
                mma_e4m3(sc, *reinterpret_cast<const uint32_t*>(&q_s[h * 8 + g][kk * 32 + tig * 4]),
                         *reinterpret_cast<const uint32_t*>(&q_s[h * 8 + g][kk * 32 + 16 + tig * 4]),
                         *reinterpret_cast<const uint32_t*>(ks + lugar(fila, byte)),
                         *reinterpret_cast<const uint32_t*>(ks + lugar(fila, byte + 16)));
            }
            const int t = t0 + nb * 8 + tig * 2;
            const float p0 = t < fin ? exp2_rapido(fminf(sc[0] * escala - c_ref, 100.f)) : 0.f;
            const float p1 = t + 1 < fin ? exp2_rapido(fminf(sc[1] * escala - c_ref, 100.f)) : 0.f;
            l += p0 + p1;
            pa[nb] = f32x2_a_bf16x2(p0, p1);
        }
        __syncthreads();   // vb completo para todos

        // o^T += v^T p^T: el fragmento A de v^T sale de vb con ldmatrix.trans (las 4 matrices 8x8 de un bloque de
        // 16 dimensiones: llaves 0-7 / 8-15 del warp x dimensiones 0-7 / 8-15); B = p^T, ya en registros.
        const int mi = carril >> 3, ri = carril & 7;
        const int f_ld = llave0 + (mi >> 1) * 8 + ri;
#pragma unroll
        for (int md = 0; md < 4; ++md) {
            uint32_t a[4];
            ldmatrix_x4_trans(a, a_smem(vb + lugar_vb(f_ld, h * 8 + md * 2 + (mi & 1))));
            mma_bf16(o[md], a[0], a[1], a[2], a[3], pa[0], pa[1]);
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    // Juntar los 4 warps de cada cabeza en la memoria compartida, ya libre.
    float* red_o = reinterpret_cast<float*>(smem);            // [4 grupos de llaves][2 cabezas][8 filas][64]
    float* red_l = red_o + 4 * 2 * 8 * 64;                     // [4][2][8]
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    if (tig == 0) red_l[(kg * 2 + h) * 8 + g] = l;
#pragma unroll
    for (int md = 0; md < 4; ++md) {   // o^T[dimension][consulta] -> red_o[grupo][cabeza][consulta][dimension]
        float* d = red_o + ((kg * 2 + h) * 8 + tig * 2) * 64 + md * 16 + g;
        d[0] = o[md][0];
        d[64] = o[md][1];
        d[8] = o[md][2];
        d[64 + 8] = o[md][3];
    }
    __syncthreads();
#pragma unroll 1
    for (int i = hilo; i < 2 * 8 * 64; i += HILOS) {
        const int hh = i / 512, fila = (i / 64) % 8, dim = i % 64;
        float acc = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w) acc += red_o[((w * 2 + hh) * 8 + fila) * 64 + dim];
        const int64_t donde = (static_cast<int64_t>(b * 2 + hh) * n_pedazos + s) * bloque_m + fila;
        ap[donde * 64 + dim] = acc;
        if (dim == 0) {
            float suma = 0.f;
#pragma unroll
            for (int w = 0; w < 4; ++w) suma += red_l[(w * 2 + hh) * 8 + fila];
            lp[donde] = suma;

        }
    }
}

// --- verify (q_len = 8): 64 filas de consulta por cabeza kv (8 piezas x 8 cabezas) --------------------------------------
// Cada warp toma una cabeza kv (h) y 16 filas REALES (grupo rg: las piezas 2rg y 2rg+1), y todas las 64 llaves de cada
// vuelta: no hay suma entre warps, y p pasa del fragmento C de q.k al fragmento A de p.v en registros (como FA2). Los
// fragmentos B de v salen de vb con ldmatrix.trans. Fila r del pedazo = pieza r / 8, cabeza de consulta h * 8 + r % 8.
template <int PAGINA>
__global__ void __launch_bounds__(HILOS, BLOQUES_POR_SM)
parcial_completo_verify(const __nv_bfloat16* __restrict__ qkv, int64_t paso_qkv, const int* __restrict__ seq_lens,
                        const int* __restrict__ tabla, int64_t paso_tabla, int pagina, const uint8_t* __restrict__ kc,
                        const uint8_t* __restrict__ vc, const float* __restrict__ k_scale, float qk_escala,
                        float escala_log2, float c_ref, float* __restrict__ lp,
                        float* __restrict__ ap, int chunk, int n_pedazos, int bloque_m) {
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(16) uint8_t q_s[8][16][64];        // [pieza][cabeza de consulta][dimension], E4M3
    uint8_t* vb = smem + ETAPAS * BLOQUE_ETAPA;
    const int s = blockIdx.x, b = blockIdx.y;
    const int hilo = threadIdx.x, warp = hilo >> 5, carril = hilo & 31;
    const int g = carril >> 2, tig = carril & 3;
    const int h = warp >> 2, rg = warp & 3;

    const int seq = seq_lens[b];
    const int inicio = s * chunk;
    const int fin = min(inicio + chunk, seq);
    const int n_vueltas = fin > inicio ? (fin - inicio + LLAVES - 1) / LLAVES : 0;

    const int fila_c = hilo >> 2, cuarto = hilo & 3;
    auto pedir = [&](int vuelta, int fila32) {
        const int64_t fila = fila32;
        uint8_t* base = smem + (vuelta % ETAPAS) * BLOQUE_ETAPA;
        const uint8_t* k = kc + fila * FILA + cuarto * 32;
        const uint8_t* v = vc + fila * FILA + cuarto * 32;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int byte = cuarto * 32 + c * 16;
            cp_async16(a_smem(base + lugar(fila_c, byte)), k + c * 16);
            cp_async16(a_smem(base + LLAVES * FILA + lugar(fila_c, byte)), v + c * 16);
        }
    };
    auto fila_de = [&](int vuelta) -> int {
        const int t = inicio + vuelta * LLAVES + fila_c;
        return t < fin ? fila_cache<PAGINA>(tabla, paso_tabla, b, t, pagina) : 0;
    };

    int fila_sig = 0;
#pragma unroll
    for (int v = 0; v < ETAPAS - 1; ++v) {
        if (v < n_vueltas) pedir(v, fila_de(v));
        cp_async_commit();
    }
    if (ETAPAS - 1 < n_vueltas) fila_sig = fila_de(ETAPAS - 1);

    {   // RMSNorm de las 8 x 16 filas de consulta en E4M3: 2 filas por hilo (8 hilos por fila, 8 valores cada uno).
#pragma unroll 1
        for (int vez = 0; vez < 4; ++vez) {
            const int fila = vez * 32 + (hilo >> 3), parte = (hilo & 7) * 8;   // fila = pieza * 16 + cabeza
            const __nv_bfloat16* fila_q = qkv + (b * 8 + (fila >> 4)) * paso_qkv + (fila & 15) * 64 + parte;
            float x[8], ss = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) { x[i] = __bfloat162float(fila_q[i]); ss += x[i] * x[i]; }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            const float inv = rsqrtf(ss / 64.f + 1e-5f);
#pragma unroll
            for (int i = 0; i < 8; ++i)
                q_s[fila >> 4][fila & 15][parte + i] =
                    static_cast<uint8_t>(__nv_cvt_float_to_fp8(x[i] * inv, __NV_SATFINITE, __NV_E4M3));
        }
    }
    __syncthreads();
    const float escala = k_scale[0] * escala_log2 * qk_escala;
    // Las filas g y g + 8 de este warp: pieza 2rg (y 2rg + 1), cabeza de consulta h * 8 + g.
    const uint8_t* q_g = &q_s[2 * rg][h * 8 + g][0];
    const uint8_t* q_g8 = &q_s[2 * rg + 1][h * 8 + g][0];

    float o[8][4];      // acumulado p.v: 8 bloques de 8 dimensiones (filas g y g + 8, las dos reales)
    float l0 = 0.f, l8 = 0.f;
#pragma unroll
    for (int nd = 0; nd < 8; ++nd)
#pragma unroll
        for (int i = 0; i < 4; ++i) o[nd][i] = 0.f;

#pragma unroll 1
    for (int vuelta = 0; vuelta < n_vueltas; ++vuelta) {
        cp_async_wait<ETAPAS - 2>();
        __syncthreads();
        if (vuelta + ETAPAS - 1 < n_vueltas) pedir(vuelta + ETAPAS - 1, fila_sig);
        cp_async_commit();
        if (vuelta + ETAPAS < n_vueltas) fila_sig = fila_de(vuelta + ETAPAS);

        const uint8_t* ks = smem + (vuelta % ETAPAS) * BLOQUE_ETAPA;
        const uint8_t* vs = ks + LLAVES * FILA;
#pragma unroll
        for (int c = 0; c < 2; ++c) {   // v E4M3 -> bf16, como en decode
            const uint4 crudo = *reinterpret_cast<const uint4*>(vs + lugar(fila_c, cuarto * 32 + c * 16));
            const uint32_t pal[4] = {crudo.x, crudo.y, crudo.z, crudo.w};
            uint32_t sal[8];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                sal[2 * w] = e4m3x2_a_bf16x2(static_cast<uint16_t>(pal[w] & 0xffffu));
                sal[2 * w + 1] = e4m3x2_a_bf16x2(static_cast<uint16_t>(pal[w] >> 16));
            }
            const int trozo = (cuarto * 32 + c * 16) / 8;
            *reinterpret_cast<uint4*>(vb + lugar_vb(fila_c, trozo)) = make_uint4(sal[0], sal[1], sal[2], sal[3]);
            *reinterpret_cast<uint4*>(vb + lugar_vb(fila_c, trozo + 1)) = make_uint4(sal[4], sal[5], sal[6], sal[7]);
        }
        __syncthreads();   // vb completo

        const int t_vuelta = inicio + vuelta * LLAVES;
        const int mi = carril >> 3, ri = carril & 7;
#pragma unroll 1
        for (int k16 = 0; k16 < 4; ++k16) {   // 16 llaves por paso: q.k (2 bloques de 8), p, y p.v
            float sc[2][4];
#pragma unroll
            for (int nb = 0; nb < 2; ++nb) {
                sc[nb][0] = sc[nb][1] = sc[nb][2] = sc[nb][3] = 0.f;
                const int fila = k16 * 16 + nb * 8 + g;
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    const int d0 = kk * 32 + tig * 4;
                    const int byte = h * 64 + d0;
                    mma_e4m3_4(sc[nb], *reinterpret_cast<const uint32_t*>(q_g + d0),
                               *reinterpret_cast<const uint32_t*>(q_g8 + d0),
                               *reinterpret_cast<const uint32_t*>(q_g + d0 + 16),
                               *reinterpret_cast<const uint32_t*>(q_g8 + d0 + 16),
                               *reinterpret_cast<const uint32_t*>(ks + lugar(fila, byte)),
                               *reinterpret_cast<const uint32_t*>(ks + lugar(fila, byte + 16)));
                }
            }
            uint32_t pa[4];   // fragmento A de p (16 filas x 16 llaves): a0 = (g, nb0), a1 = (g+8, nb0), a2/a3 = nb1
#pragma unroll
            for (int nb = 0; nb < 2; ++nb) {
                const int t = t_vuelta + k16 * 16 + nb * 8 + tig * 2;
                const bool v0 = t < fin, v1 = t + 1 < fin;
                const float p0 = v0 ? exp2_rapido(fminf(sc[nb][0] * escala - c_ref, 100.f)) : 0.f;
                const float p1 = v1 ? exp2_rapido(fminf(sc[nb][1] * escala - c_ref, 100.f)) : 0.f;
                const float p2 = v0 ? exp2_rapido(fminf(sc[nb][2] * escala - c_ref, 100.f)) : 0.f;
                const float p3 = v1 ? exp2_rapido(fminf(sc[nb][3] * escala - c_ref, 100.f)) : 0.f;
                l0 += p0 + p1;
                l8 += p2 + p3;
                pa[nb * 2] = f32x2_a_bf16x2(p0, p1);
                pa[nb * 2 + 1] = f32x2_a_bf16x2(p2, p3);
            }
            // B de v (16 llaves x 8 dimensiones) con ldmatrix.trans: matrices (llaves 0-7 | 8-15) x (bloque nd | nd+1).
#pragma unroll
            for (int nd = 0; nd < 8; nd += 2) {
                const int f_ld = k16 * 16 + (mi & 1) * 8 + ri;
                uint32_t bb[4];
                ldmatrix_x4_trans(bb, a_smem(vb + lugar_vb(f_ld, h * 8 + nd + (mi >> 1))));
                mma_bf16(o[nd], pa[0], pa[1], pa[2], pa[3], bb[0], bb[1]);
                mma_bf16(o[nd + 1], pa[0], pa[1], pa[2], pa[3], bb[2], bb[3]);
            }
        }
    }
    cp_async_wait<0>();

    // Sin suma entre warps: cada warp vio todas las llaves para sus 16 filas.
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l8 += __shfl_xor_sync(0xffffffffu, l8, 1);
    l8 += __shfl_xor_sync(0xffffffffu, l8, 2);
    const int r0 = rg * 16 + g;   // fila del pedazo (pieza 2rg, cabeza g) y r0 + 8 (pieza 2rg + 1)
    const int64_t base = (static_cast<int64_t>(b * 2 + h) * n_pedazos + s) * bloque_m;
#pragma unroll
    for (int nd = 0; nd < 8; ++nd) {
        *reinterpret_cast<float2*>(&ap[(base + r0) * 64 + nd * 8 + tig * 2]) = make_float2(o[nd][0], o[nd][1]);
        *reinterpret_cast<float2*>(&ap[(base + r0 + 8) * 64 + nd * 8 + tig * 2]) = make_float2(o[nd][2], o[nd][3]);
    }
    if (tig == 0) {
        lp[base + r0] = l0;
        lp[base + r0 + 8] = l8;


    }
}

}  // namespace muse
"""

_CUDA_ANFITRION = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <array>
#include <mutex>

template <int PAGINA, bool VERIFY>
void lanzar_nativo_v16(const torch::Tensor& qkv, const torch::Tensor& seq_lens,
                       const torch::Tensor& tabla, const torch::Tensor& kc, const torch::Tensor& vc,
                       const torch::Tensor& k_scale, const torch::Tensor& lp, const torch::Tensor& ap,
                       int64_t chunk, int64_t n_pedazos, int64_t bloque_m,
                       double qk_escala, double escala_log2, double c_ref) {
    const c10::cuda::CUDAGuard guard(qkv.device());
    const int memoria = muse::ETAPAS * muse::BLOQUE_ETAPA + muse::LLAVES * muse::FILA_VB;
    const int dispositivo = qkv.get_device();
    TORCH_CHECK(dispositivo >= 0 && dispositivo < 64, "dispositivo CUDA fuera de rango");
    static std::array<std::once_flag, 64> inicializados;
    std::call_once(inicializados[dispositivo], [=]() {
        if constexpr (VERIFY) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(muse::parcial_completo_verify<PAGINA>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, memoria));
        } else {
            C10_CUDA_CHECK(cudaFuncSetAttribute(muse::parcial_completo<PAGINA>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, memoria));
        }
    });
    dim3 malla(static_cast<unsigned>(n_pedazos), static_cast<unsigned>(seq_lens.size(0)));
    const auto stream = at::cuda::getCurrentCUDAStream(dispositivo);
    if constexpr (VERIFY) {
        muse::parcial_completo_verify<PAGINA><<<malla, muse::HILOS, memoria, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), qkv.stride(0), seq_lens.data_ptr<int>(),
            tabla.data_ptr<int>(), tabla.stride(0), PAGINA,
            reinterpret_cast<const uint8_t*>(kc.data_ptr()), reinterpret_cast<const uint8_t*>(vc.data_ptr()),
            k_scale.data_ptr<float>(), static_cast<float>(qk_escala), static_cast<float>(escala_log2),
            static_cast<float>(c_ref), lp.data_ptr<float>(), ap.data_ptr<float>(),
            static_cast<int>(chunk), static_cast<int>(n_pedazos), static_cast<int>(bloque_m));
    } else {
        muse::parcial_completo<PAGINA><<<malla, muse::HILOS, memoria, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), qkv.stride(0), seq_lens.data_ptr<int>(),
            tabla.data_ptr<int>(), tabla.stride(0), PAGINA,
            reinterpret_cast<const uint8_t*>(kc.data_ptr()), reinterpret_cast<const uint8_t*>(vc.data_ptr()),
            k_scale.data_ptr<float>(), static_cast<float>(qk_escala), static_cast<float>(escala_log2),
            static_cast<float>(c_ref), lp.data_ptr<float>(), ap.data_ptr<float>(),
            static_cast<int>(chunk), static_cast<int>(n_pedazos), static_cast<int>(bloque_m));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void parcial_completo_lanzar(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor tabla,
                             torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,
                             torch::Tensor lp, torch::Tensor ap, int64_t pagina, int64_t chunk,
                             int64_t n_pedazos, int64_t bloque_m,
                             double qk_escala, double escala_log2, double c_ref) {
    TORCH_CHECK(pagina == 1 || pagina == 64, "pagina CUDA debe ser 1 o 64");
    if (pagina == 1) {
        lanzar_nativo_v16<1, false>(qkv, seq_lens, tabla, kc, vc, k_scale, lp, ap,
                                    chunk, n_pedazos, bloque_m, qk_escala, escala_log2, c_ref);
    } else {
        lanzar_nativo_v16<64, false>(qkv, seq_lens, tabla, kc, vc, k_scale, lp, ap,
                                     chunk, n_pedazos, bloque_m, qk_escala, escala_log2, c_ref);
    }
}

void parcial_verify_lanzar(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor tabla,
                           torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,
                           torch::Tensor lp, torch::Tensor ap, int64_t pagina, int64_t chunk,
                           int64_t n_pedazos, int64_t bloque_m,
                           double qk_escala, double escala_log2, double c_ref) {
    TORCH_CHECK(pagina == 1 || pagina == 64, "pagina CUDA debe ser 1 o 64");
    if (pagina == 1) {
        lanzar_nativo_v16<1, true>(qkv, seq_lens, tabla, kc, vc, k_scale, lp, ap,
                                   chunk, n_pedazos, bloque_m, qk_escala, escala_log2, c_ref);
    } else {
        lanzar_nativo_v16<64, true>(qkv, seq_lens, tabla, kc, vc, k_scale, lp, ap,
                                    chunk, n_pedazos, bloque_m, qk_escala, escala_log2, c_ref);
    }
}
"""

_CUDA = {}


def _modulo_cuda():
    """Compila v16 una vez; si falla, queda la ruta Triton de referencia fija."""
    if "modulo" not in _CUDA:
        try:
            from torch.utils.cpp_extension import load_inline
            _CUDA["modulo"] = load_inline(
                name="muse_parcial_fijo_v16", cpp_sources='void parcial_completo_lanzar(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor tabla,\n                             torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,\n                             torch::Tensor lp, torch::Tensor ap, int64_t pagina, int64_t chunk,\n                             int64_t n_pedazos, int64_t bloque_m,\n                             double qk_escala, double escala_log2, double c_ref);\nvoid parcial_verify_lanzar(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor tabla,\n                           torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,\n                           torch::Tensor lp, torch::Tensor ap, int64_t pagina, int64_t chunk,\n                           int64_t n_pedazos, int64_t bloque_m,\n                           double qk_escala, double escala_log2, double c_ref);',
                cuda_sources=_CUDA_FUENTE + _CUDA_ANFITRION,
                functions=["parcial_completo_lanzar", "parcial_verify_lanzar"],
                extra_cflags=["-O3", "-std=c++20"],
                extra_cuda_cflags=["-O3", "-std=c++20"], verbose=False)
            print("v16: kernel CUDA compilado (paginas 1/64, sin MP)", flush=True)
        except Exception as error:
            print(f"v16: el kernel CUDA no compilo, sigo con Triton fijo: {type(error).__name__}: {error}"[:4000], flush=True)
            _CUDA["modulo"] = None
    return _CUDA["modulo"]


@triton.jit
def _parcial_fijo(QKV, SEQ, TABLA, KC, VC, KS, ROPE, LP, AP, paso_qkv, paso_tabla, chunk,
                  n_pedazos, qk_escala, escala_log2, c_ref, HQ: tl.constexpr, HKV: tl.constexpr,
                  L: tl.constexpr, G: tl.constexpr, PAGINA: tl.constexpr, VENTANA: tl.constexpr,
                  CON_ROPE: tl.constexpr, BLOQUE_M: tl.constexpr, BLOQUE_N: tl.constexpr,
                  EN_CPU: tl.constexpr):
    """Split-K aditivo: misma referencia para todos los pedazos; sin maximos ni MP."""
    s = tl.program_id(0)
    bh = tl.program_id(1)
    b, h = bh // HKV, bh % HKV
    seq = tl.load(SEQ + b)
    filas = tl.arange(0, BLOQUE_M)
    fila_ok = filas < L * G
    j = tl.where(fila_ok, filas // G, 0)
    cabeza = h * G + filas % G
    pos = seq + j
    d = tl.arange(0, 64)
    base_q = QKV + (b * L + j).to(tl.int64) * paso_qkv + cabeza * 64
    # Conserva las consultas BF16 del camino Triton v15: cambia la reduccion,
    # no la precision de Q. La ruta CUDA sigue usando Q E4M3, como v15/FA3.
    q = _norma_rope(base_q, pos, ROPE, qk_escala, CON_ROPE).to(tl.bfloat16)
    lo = tl.maximum(seq - VENTANA + 1, 0) if VENTANA > 0 else seq * 0
    inicio = lo + s * chunk
    fin = tl.minimum(inicio + chunk, seq)
    escala = tl.load(KS) * escala_log2
    suma = tl.zeros([BLOQUE_M], tl.float32)
    acc = tl.zeros([BLOQUE_M, 64], tl.float32)
    fila_sig = _filas(TABLA, b, paso_tabla, inicio + tl.arange(0, BLOQUE_N), fin, PAGINA)
    for t0 in tl.range(inicio, fin, BLOQUE_N, num_stages=2):
        t = t0 + tl.arange(0, BLOQUE_N)
        ok = t < fin
        fila_c = fila_sig
        fila_sig = _filas(TABLA, b, paso_tabla, t + BLOQUE_N, fin, PAGINA)
        off = fila_c[:, None] * (HKV * 64) + h * 64 + d[None, :]
        k = tl.load(KC + off, mask=ok[:, None], other=0.0)
        v = tl.load(VC + off, mask=ok[:, None], other=0.0)
        valido = ok[None, :]
        if VENTANA > 0:
            valido = valido & (t[None, :] > pos[:, None] - VENTANA)
        suma, acc = _sumar(q, k, v, valido, suma, acc, escala, c_ref, False, EN_CPU)
    donde = (bh * n_pedazos + s) * BLOQUE_M + filas
    tl.store(LP + donde, suma, mask=fila_ok)
    tl.store(AP + donde[:, None] * 64 + d[None, :], acc, mask=fila_ok[:, None])


@triton.jit
def _final_fijo(QKV, SEQ, LOC, KC, VC, KS, VS, ROPE, LP, AP, OUT, paso_qkv,
                n_pedazos: tl.constexpr, qk_escala, escala_log2, c_ref,
                HQ: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr, G: tl.constexpr,
                VENTANA: tl.constexpr, CON_ROPE: tl.constexpr, BLOQUE_M: tl.constexpr,
                BLOQUE_S: tl.constexpr, BLOQUE_L: tl.constexpr):
    """Suma los pedazos directamente y agrega TODOS los tokens nuevos causales."""
    pid = tl.program_id(0)
    bh, r = pid // (L * G), pid % (L * G)
    b, h = bh // HKV, bh % HKV
    j = r // G
    cabeza = h * G + r % G
    seq = tl.load(SEQ + b)
    d = tl.arange(0, 64)
    uno = tl.arange(0, 1)
    ks, vs = tl.load(KS), tl.load(VS)
    suma = tl.full((), 0.0, tl.float32)
    acc = tl.zeros([64], tl.float32)
    for s0 in range(0, n_pedazos, BLOQUE_S):
        sidx = s0 + tl.arange(0, BLOQUE_S)
        hay = sidx < n_pedazos
        donde = (bh * n_pedazos + sidx) * BLOQUE_M + r
        suma = suma + tl.sum(tl.load(LP + donde, mask=hay, other=0.0), 0)
        acc = acc + tl.sum(tl.load(AP + donde[:, None] * 64 + d[None, :],
                                   mask=hay[:, None], other=0.0), 0)
    fila_q = QKV + (b * L + j).to(tl.int64) * paso_qkv + cabeza * 64 + uno
    q = tl.reshape(_norma_rope(fila_q, seq + j + uno, ROPE, qk_escala, CON_ROPE), (64,))
    q = q.to(tl.bfloat16).to(tl.float32)
    li = tl.arange(0, BLOQUE_L)
    hay_l = li < L
    i_c = tl.where(hay_l, li, 0)
    tok = (b * L + i_c).to(tl.int64)
    k8 = _a_fp8(_norma_rope(QKV + tok * paso_qkv + (HQ + h) * 64,
                           seq + i_c, ROPE, 1.0, CON_ROPE) / ks)
    v8 = _a_fp8(tl.load(QKV + tok[:, None] * paso_qkv + (HQ + HKV + h) * 64 + d[None, :]).to(tl.float32) / vs)
    destino = tl.load(LOC + tok)[:, None] * (HKV * 64) + h * 64 + d[None, :]
    escribe = (r == 0) & hay_l[:, None]
    tl.store(KC + destino, k8, mask=escribe)
    tl.store(VC + destino, v8, mask=escribe)
    sc = tl.sum(q[None, :] * k8.to(tl.float32), 1) * ks * escala_log2
    valido = hay_l & (li <= j)
    if VENTANA > 0:
        valido = valido & (li > j - VENTANA)
    p = tl.math.exp2(tl.where(valido, tl.minimum(sc - c_ref, TOPE), float("-inf")))
    suma = suma + tl.sum(p, 0)
    acc = acc + tl.sum(p[:, None] * v8.to(tl.float32), 0)
    # La division y epsilon se conservan: RMSNorm no es exactamente invariante
    # a escala cuando epsilon es distinto de cero.
    o = acc / suma * vs
    o = o * tl.rsqrt(tl.sum(o * o, 0) / 64 + 1e-5)
    puerta = tl.load(QKV + (b * L + j).to(tl.int64) * paso_qkv +
                     (HQ + 2 * HKV + cabeza) * 64 + d).to(tl.float32)
    o = o * tl.sigmoid(puerta)
    tl.store(OUT + (b * L + j).to(tl.int64) * (HQ * 64) + cabeza * 64 + d, o.to(tl.bfloat16))


def _metodo_pedazos_referencia(data):
    """Respaldo con maximos para configuraciones fuera de la cota de referencia fija."""
    cfg = data["config"]
    batch, largo = cfg["batch"], cfg["q_len"]
    hq, hkv, dim, pagina = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    assert dim == 64 and hq % hkv == 0
    g = hq // hkv
    ventana = cfg["window"]
    qkv = data["qkv"].reshape(batch * largo, -1)
    tabla, device = data["page_table"], qkv.device
    con_rope = bool(cfg["rope_theta"])
    rope = _cos_sin(cfg["rope_theta"], dim, device) if con_rope else qkv
    qk_escala = float(cfg["qk_scale_factor"])
    escala_log2 = float(cfg["sm_scale"]) * LOG2E
    max_llaves = max(1, ventana - 1 if ventana else tabla.shape[1] * pagina)
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
    _parcial_v2[(n_pedazos, n_bh)](
        qkv, data["seq_lens"], tabla, data["k_cache"], data["v_cache"], data["k_scale"], rope,
        mp, lp, ap, qkv.stride(0), tabla.stride(0), chunk, n_pedazos, qk_escala, escala_log2,
        HQ=hq, HKV=hkv, L=largo, G=g, PAGINA=pagina, VENTANA=ventana, CON_ROPE=con_rope,
        BLOQUE_M=bloque_m, BLOQUE_N=bloque_n, EN_CPU=device.type == "cpu", num_warps=4, num_stages=2)
    _final_v2[(n_bh * largo * g,)](
        qkv, data["seq_lens"], data["cache_loc"], data["k_cache"], data["v_cache"], data["k_scale"],
        data["v_scale"], rope, mp, lp, ap, out, qkv.stride(0), n_pedazos, qk_escala, escala_log2,
        HQ=hq, HKV=hkv, L=largo, G=g, CON_ROPE=con_rope, BLOQUE_M=bloque_m,
        BLOQUE_S=min(BLOQUE_S_V2, triton.next_power_of_2(n_pedazos)),
        BLOQUE_L=triton.next_power_of_2(largo), num_warps=4)
    return out


def _metodo_pedazos(data):
    """v16: referencia fija en ambos kernels, sin tensor de maximos parciales."""
    cfg = data["config"]
    batch, largo = cfg["batch"], cfg["q_len"]
    hq, hkv, dim, pagina = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    assert dim == 64 and hq % hkv == 0
    g = hq // hkv
    qk_escala = float(cfg["qk_scale_factor"])
    escala_log2 = float(cfg["sm_scale"]) * LOG2E
    cota = qk_escala * dim * escala_log2
    # La forma de referencia estable queda disponible para escalas no previstas.
    # Este control usa SOLO config, nunca valores de tensores.
    if not (0.0 < cota <= 75.0):
        return _metodo_pedazos_referencia(data)
    c_ref = cota - HOLGURA
    ventana = cfg["window"]
    qkv = data["qkv"].reshape(batch * largo, -1)
    tabla, device = data["page_table"], qkv.device
    con_rope = bool(cfg["rope_theta"])
    rope = _cos_sin(cfg["rope_theta"], dim, device) if con_rope else qkv
    max_llaves = max(1, ventana - 1 if ventana else tabla.shape[1] * pagina)
    bloque_n = 64 if max_llaves <= 4096 else 128
    n_bh = batch * hkv
    n_pedazos = max(1, min(triton.cdiv(PROGRAMAS_POR_SM * _n_sms(device), n_bh),
                           triton.cdiv(max_llaves, 2 * bloque_n)))
    chunk = triton.cdiv(triton.cdiv(max_llaves, n_pedazos), bloque_n) * bloque_n
    n_pedazos = max(1, triton.cdiv(max_llaves, chunk))
    bloque_m = max(16, triton.next_power_of_2(largo * g))
    lp = torch.empty((n_bh * n_pedazos * bloque_m,), dtype=torch.float32, device=device)
    ap = torch.empty((n_bh * n_pedazos * bloque_m, 64), dtype=torch.float32, device=device)
    out = torch.empty((batch, largo, hq * 64), dtype=torch.bfloat16, device=device)
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    usa_cuda = (not ventana and not con_rope and largo in (1, 8) and hq == 16 and hkv == 2
                and pagina in (1, 64) and device.type == "cuda")
    modulo = _modulo_cuda() if usa_cuda else None
    if modulo is not None:
        lanzar = modulo.parcial_completo_lanzar if largo == 1 else modulo.parcial_verify_lanzar
        lanzar(qkv, data["seq_lens"], tabla, k_cache, v_cache, data["k_scale"], lp, ap,
               pagina, chunk, n_pedazos, bloque_m, qk_escala, escala_log2, c_ref)
    else:
        _parcial_fijo[(n_pedazos, n_bh)](
            qkv, data["seq_lens"], tabla, k_cache, v_cache, data["k_scale"], rope, lp, ap,
            qkv.stride(0), tabla.stride(0), chunk, n_pedazos, qk_escala, escala_log2, c_ref,
            HQ=hq, HKV=hkv, L=largo, G=g, PAGINA=pagina, VENTANA=ventana, CON_ROPE=con_rope,
            BLOQUE_M=bloque_m, BLOQUE_N=bloque_n, EN_CPU=device.type == "cpu", num_warps=4, num_stages=2)
    _final_fijo[(n_bh * largo * g,)](
        qkv, data["seq_lens"], data["cache_loc"], k_cache, v_cache, data["k_scale"], data["v_scale"],
        rope, lp, ap, out, qkv.stride(0), n_pedazos, qk_escala, escala_log2, c_ref,
        HQ=hq, HKV=hkv, L=largo, G=g, VENTANA=ventana, CON_ROPE=con_rope, BLOQUE_M=bloque_m,
        BLOQUE_S=min(BLOQUE_S_V2, triton.next_power_of_2(n_pedazos)),
        BLOQUE_L=triton.next_power_of_2(largo), num_warps=4)
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


def _kernel_v16(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, largo, completo = cfg["batch"], cfg["q_len"], not cfg["window"]
    l = 1 if largo == 1 else 8
    if completo:
        if batch <= 4:   # v15: verify 16 completo pasa al kernel CUDA de pedazos (antes quorum, 252 us)
            return _metodo_quorum(data, CONFIG[(l, True)])
        return _metodo_pedazos(data)
    if l == 1:
        return _metodo_quorum(data, QUORUM_VENTANA_DECODE) if batch <= 4 else _metodo_pedazos(data)
    return _metodo_quorum(data, QUORUM_VENTANA_VERIFY) if batch >= 24 else _metodo_pedazos(data)


# v17: sustituir la ruta verify/full por FA3 con pre/postprocesado fusionado.
# No se importa codigo del juez ni de los archivos baseline/reference.
_FA3_V17 = {}


def _cargar_fa3_v17():
    """Solo guarda la API de una biblioteca instalada; nunca datos ni salidas."""
    if "api" not in _FA3_V17:
        import importlib
        import inspect
        api = None
        motivos = []
        for nombre in ("sgl_kernel.flash_attn", "sglang.kernels.ops.attention.flash_attention"):
            try:
                fn = getattr(importlib.import_module(nombre), "flash_attn_with_kvcache")
                parametros = inspect.signature(fn).parameters
                pagina = "page_table" if "page_table" in parametros else "block_table"
                escalas = []
                for eje in ("q", "k", "v"):
                    candidatos = (eje + "_descale", "descale_" + eje)
                    escalas.append(next((p for p in candidatos if p in parametros), None))
                necesarios = {"cache_seqlens", "softmax_scale", "causal", pagina}
                if not necesarios.issubset(parametros) or any(p is None for p in escalas):
                    motivos.append(nombre + ": API sin soporte explicito de paginacion y escalas FP8")
                    continue
                api = (fn, pagina, tuple(escalas), frozenset(parametros))
                print("v17: verify/full usa FA3 desde " + nombre, flush=True)
                break
            except (ImportError, AttributeError, TypeError, ValueError) as error:
                motivos.append(nombre + ": " + type(error).__name__)
        if api is None:
            print("v17: FA3 compatible no disponible; mantengo v16. " + "; ".join(motivos), flush=True)
        _FA3_V17["api"] = api
    return _FA3_V17["api"]


@triton.jit
def _preparar_fa3_v17(QKV, LOC, KC, VC, KS, VS, Q8, QD, KD, VD, paso_qkv, qk_escala,
                      HQ: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr,
                      G: tl.constexpr, GG: tl.constexpr):
    """Una CTA por token/cabeza KV: QK-norm, Q FP8, cache FP8 y escalas."""
    tok = tl.program_id(0)
    h = tl.program_id(1)
    b, j = tok // L, tok % L
    g = tl.arange(0, GG)
    hay = g < G
    cabeza = h * G + tl.where(hay, g, 0)
    base = QKV + tok.to(tl.int64) * paso_qkv
    pos_q = tl.full([GG], 0, tl.int32)  # esta ruta no tiene RoPE
    q = _norma_rope(base + cabeza * 64, pos_q, QKV, qk_escala, False)
    # QK-norm produce BF16 antes del redondeo a E4M3, como la ruta SGLang.
    q8 = _a_fp8(q.to(tl.bfloat16).to(tl.float32))
    d = tl.arange(0, 64)
    tl.store(Q8 + (tok.to(tl.int64) * HQ + cabeza[:, None]) * 64 + d[None, :],
             q8, mask=hay[:, None])
    uno = tl.arange(0, 1)
    ks, vs = tl.load(KS), tl.load(VS)
    k8 = _a_fp8(_norma_rope(base + (HQ + h) * 64 + uno, uno, QKV, 1.0, False) / ks)
    v8 = _a_fp8(tl.load(base + (HQ + HKV + h) * 64 + d).to(tl.float32) / vs)
    fila = tl.load(LOC + tok).to(tl.int64)
    tl.store(KC + fila * (HKV * 64) + h * 64 + d, tl.reshape(k8, (64,)))
    tl.store(VC + fila * (HKV * 64) + h * 64 + d, v8)
    if j == 0:
        tl.store(QD + b * HKV + h, 1.0)
        tl.store(KD + b * HKV + h, ks)
        tl.store(VD + b * HKV + h, vs)


@triton.jit
def _salida_fa3_v17(A, QKV, OUT, paso_a0, paso_a1, paso_qkv,
                    HQ: tl.constexpr, HKV: tl.constexpr):
    pid = tl.program_id(0)
    tok, cabeza = pid // HQ, pid % HQ
    d = tl.arange(0, 64)
    a = tl.load(A + tok.to(tl.int64) * paso_a0 + cabeza * paso_a1 + d).to(tl.float32)
    a = a * tl.rsqrt(tl.sum(a * a, 0) / 64 + 1e-5)
    gate = tl.load(QKV + tok.to(tl.int64) * paso_qkv +
                   (HQ + 2 * HKV + cabeza) * 64 + d).to(tl.float32)
    tl.store(OUT + (tok.to(tl.int64) * HQ + cabeza) * 64 + d,
             (a * tl.sigmoid(gate)).to(tl.bfloat16))


def _metodo_fa3_v17(data, api):
    cfg = data["config"]
    batch, largo = cfg["batch"], cfg["q_len"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    g = hq // hkv
    qkv = data["qkv"].reshape(batch * largo, -1)
    device = qkv.device
    q8 = torch.empty((batch, largo, hq, 64), dtype=torch.float8_e4m3fn, device=device)
    # Las escalas reales de los tensores se copian en la misma CTA de preparacion.
    # No se asume que sean 1 ni se leen sus valores en CPU.
    qd = torch.empty((batch, hkv), dtype=torch.float32, device=device)
    kd, vd = torch.empty_like(qd), torch.empty_like(qd)
    _preparar_fa3_v17[(batch * largo, hkv)](
        qkv, data["cache_loc"], data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
        q8, qd, kd, vd, qkv.stride(0), float(cfg["qk_scale_factor"]),
        HQ=hq, HKV=hkv, L=largo, G=g, GG=triton.next_power_of_2(g), num_warps=4)
    fn, pagina_kw, escala_kw, parametros = api
    kwargs = {
        "cache_seqlens": data["cache_seqlens"],
        pagina_kw: data["page_table"],
        escala_kw[0]: qd,
        escala_kw[1]: kd,
        escala_kw[2]: vd,
        "softmax_scale": float(cfg["sm_scale"]),
        "causal": True,
    }
    if "window_size" in parametros:
        kwargs["window_size"] = (-1, -1)
    if "num_splits" in parametros:
        kwargs["num_splits"] = 0  # planificador normal de FA3
    if "scheduler_metadata" in parametros:
        kwargs["scheduler_metadata"] = data.get("scheduler_metadata")
    if "return_softmax_lse" in parametros:
        kwargs["return_softmax_lse"] = False
    resultado = fn(q8, data["k_cache"], data["v_cache"], **kwargs)
    a = resultado[0] if isinstance(resultado, (tuple, list)) else resultado
    a = a.reshape(batch * largo, hq, 64)
    out = torch.empty((batch, largo, hq * 64), dtype=torch.bfloat16, device=device)
    _salida_fa3_v17[(batch * largo * hq,)](
        a, qkv, out, a.stride(0), a.stride(1), qkv.stride(0), HQ=hq, HKV=hkv, num_warps=4)
    return out


# ------------------------------------------------------------
# MuseOne: optimizacion CUDA aislada en verify/full (batch >= 16).
_MUSEONE = {}
_MUSEONE_FUENTE = r"""// MuseOne CUDA nativo: variante de verify con conversiones V warp-local.
// No modifica el kernel decode de v17.
namespace muse {
__device__ __forceinline__ int lugar_vwarp(int f, int trozo) {
    return f * 128 + ((trozo ^ (f & 7)) << 4);
}
template <int PAGINA>
__global__ void __launch_bounds__(HILOS, BLOQUES_POR_SM)
parcial_verificar_museone(const __nv_bfloat16* __restrict__ qkv, int64_t paso_qkv, const int* __restrict__ seq_lens,
                        const int* __restrict__ tabla, int64_t paso_tabla, int pagina, const uint8_t* __restrict__ kc,
                        const uint8_t* __restrict__ vc, const float* __restrict__ k_scale, float qk_escala,
                        float escala_log2, float c_ref, float* __restrict__ lp,
                        float* __restrict__ ap, int chunk, int n_pedazos, int bloque_m) {
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(16) uint8_t q_s[8][16][64];        // [pieza][cabeza de consulta][dimension], E4M3
    const int s = blockIdx.x, b = blockIdx.y;
    const int hilo = threadIdx.x, warp = hilo >> 5, carril = hilo & 31;
    const int g = carril >> 2, tig = carril & 3;
    const int h = warp >> 2, rg = warp & 3;
    uint8_t* vb_warp = smem + ETAPAS * BLOQUE_ETAPA + warp * 16 * 128;

    const int seq = seq_lens[b];
    const int inicio = s * chunk;
    const int fin = min(inicio + chunk, seq);
    const int n_vueltas = fin > inicio ? (fin - inicio + LLAVES - 1) / LLAVES : 0;
    if (n_vueltas == 0) return;  // Nunca escribir pedazos sin tokens.

    const int fila_c = hilo >> 2, cuarto = hilo & 3;
    auto pedir = [&](int vuelta, int fila32) {
        const int64_t fila = fila32;
        uint8_t* base = smem + (vuelta % ETAPAS) * BLOQUE_ETAPA;
        const uint8_t* k = kc + fila * FILA + cuarto * 32;
        const uint8_t* v = vc + fila * FILA + cuarto * 32;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int byte = cuarto * 32 + c * 16;
            cp_async16(a_smem(base + lugar(fila_c, byte)), k + c * 16);
            cp_async16(a_smem(base + LLAVES * FILA + lugar(fila_c, byte)), v + c * 16);
        }
    };
    auto fila_de = [&](int vuelta) -> int {
        const int t = inicio + vuelta * LLAVES + fila_c;
        return t < fin ? fila_cache<PAGINA>(tabla, paso_tabla, b, t, pagina) : 0;
    };

    int fila_sig = 0;
#pragma unroll
    for (int v = 0; v < ETAPAS - 1; ++v) {
        if (v < n_vueltas) pedir(v, fila_de(v));
        cp_async_commit();
    }
    if (ETAPAS - 1 < n_vueltas) fila_sig = fila_de(ETAPAS - 1);

    {   // RMSNorm de las 8 x 16 filas de consulta en E4M3: 2 filas por hilo (8 hilos por fila, 8 valores cada uno).
#pragma unroll 1
        for (int vez = 0; vez < 4; ++vez) {
            const int fila = vez * 32 + (hilo >> 3), parte = (hilo & 7) * 8;   // fila = pieza * 16 + cabeza
            const __nv_bfloat16* fila_q = qkv + (b * 8 + (fila >> 4)) * paso_qkv + (fila & 15) * 64 + parte;
            float x[8], ss = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) { x[i] = __bfloat162float(fila_q[i]); ss += x[i] * x[i]; }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            const float inv = rsqrtf(ss / 64.f + 1e-5f);
#pragma unroll
            for (int i = 0; i < 8; ++i)
                q_s[fila >> 4][fila & 15][parte + i] =
                    static_cast<uint8_t>(__nv_cvt_float_to_fp8(x[i] * inv, __NV_SATFINITE, __NV_E4M3));
        }
    }
    __syncthreads();
    const float escala = k_scale[0] * escala_log2 * qk_escala;
    // Las filas g y g + 8 de este warp: pieza 2rg (y 2rg + 1), cabeza de consulta h * 8 + g.
    const uint8_t* q_g = &q_s[2 * rg][h * 8 + g][0];
    const uint8_t* q_g8 = &q_s[2 * rg + 1][h * 8 + g][0];

    float o[8][4];      // acumulado p.v: 8 bloques de 8 dimensiones (filas g y g + 8, las dos reales)
    float l0 = 0.f, l8 = 0.f;
#pragma unroll
    for (int nd = 0; nd < 8; ++nd)
#pragma unroll
        for (int i = 0; i < 4; ++i) o[nd][i] = 0.f;

#pragma unroll 1
    for (int vuelta = 0; vuelta < n_vueltas; ++vuelta) {
        cp_async_wait<ETAPAS - 2>();
        __syncthreads();
        if (vuelta + ETAPAS - 1 < n_vueltas) pedir(vuelta + ETAPAS - 1, fila_sig);
        cp_async_commit();
        if (vuelta + ETAPAS < n_vueltas) fila_sig = fila_de(vuelta + ETAPAS);

        const uint8_t* ks = smem + (vuelta % ETAPAS) * BLOQUE_ETAPA;
        const uint8_t* vs = ks + LLAVES * FILA;
        const int t_vuelta = inicio + vuelta * LLAVES;
        const int mi = carril >> 3, ri = carril & 7;
#pragma unroll 1
        for (int k16 = 0; k16 < 4; ++k16) {   // 16 llaves por paso: q.k (2 bloques de 8), p, y p.v
            // Cada warp convierte solo sus 16 llaves para su propia cabeza KV.
            // vb_warp nunca se comparte entre warps: basta una barrera de warp.
            const int fila_v = carril >> 1, mitad_v = carril & 1;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int byte = h * 64 + mitad_v * 32 + c * 16;
                const uint4 crudo = *reinterpret_cast<const uint4*>(vs + lugar(k16 * 16 + fila_v, byte));
                const uint32_t pal[4] = {crudo.x, crudo.y, crudo.z, crudo.w};
                uint32_t sal[8];
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    sal[2 * w] = e4m3x2_a_bf16x2(static_cast<uint16_t>(pal[w] & 0xffffu));
                    sal[2 * w + 1] = e4m3x2_a_bf16x2(static_cast<uint16_t>(pal[w] >> 16));
                }
                const int col16 = mitad_v * 4 + c * 2;
                *reinterpret_cast<uint4*>(vb_warp + lugar_vwarp(fila_v, col16)) = make_uint4(sal[0], sal[1], sal[2], sal[3]);
                *reinterpret_cast<uint4*>(vb_warp + lugar_vwarp(fila_v, col16 + 1)) = make_uint4(sal[4], sal[5], sal[6], sal[7]);
            }
            __syncwarp();
            float sc[2][4];
#pragma unroll
            for (int nb = 0; nb < 2; ++nb) {
                sc[nb][0] = sc[nb][1] = sc[nb][2] = sc[nb][3] = 0.f;
                const int fila = k16 * 16 + nb * 8 + g;
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    const int d0 = kk * 32 + tig * 4;
                    const int byte = h * 64 + d0;
                    mma_e4m3_4(sc[nb], *reinterpret_cast<const uint32_t*>(q_g + d0),
                               *reinterpret_cast<const uint32_t*>(q_g8 + d0),
                               *reinterpret_cast<const uint32_t*>(q_g + d0 + 16),
                               *reinterpret_cast<const uint32_t*>(q_g8 + d0 + 16),
                               *reinterpret_cast<const uint32_t*>(ks + lugar(fila, byte)),
                               *reinterpret_cast<const uint32_t*>(ks + lugar(fila, byte + 16)));
                }
            }
            uint32_t pa[4];   // fragmento A de p (16 filas x 16 llaves): a0 = (g, nb0), a1 = (g+8, nb0), a2/a3 = nb1
#pragma unroll
            for (int nb = 0; nb < 2; ++nb) {
                const int t = t_vuelta + k16 * 16 + nb * 8 + tig * 2;
                const bool v0 = t < fin, v1 = t + 1 < fin;
                const float p0 = v0 ? exp2_rapido(fminf(sc[nb][0] * escala - c_ref, 100.f)) : 0.f;
                const float p1 = v1 ? exp2_rapido(fminf(sc[nb][1] * escala - c_ref, 100.f)) : 0.f;
                const float p2 = v0 ? exp2_rapido(fminf(sc[nb][2] * escala - c_ref, 100.f)) : 0.f;
                const float p3 = v1 ? exp2_rapido(fminf(sc[nb][3] * escala - c_ref, 100.f)) : 0.f;
                l0 += p0 + p1;
                l8 += p2 + p3;
                pa[nb * 2] = f32x2_a_bf16x2(p0, p1);
                pa[nb * 2 + 1] = f32x2_a_bf16x2(p2, p3);
            }
            // B de v (16 llaves x 8 dimensiones) con ldmatrix.trans: matrices (llaves 0-7 | 8-15) x (bloque nd | nd+1).
#pragma unroll
            for (int nd = 0; nd < 8; nd += 2) {
                const int f_ld = k16 * 16 + (mi & 1) * 8 + ri;
                uint32_t bb[4];
                ldmatrix_x4_trans(bb, a_smem(vb_warp + lugar_vwarp((mi & 1) * 8 + ri, nd + (mi >> 1))));
                mma_bf16(o[nd], pa[0], pa[1], pa[2], pa[3], bb[0], bb[1]);
                mma_bf16(o[nd + 1], pa[0], pa[1], pa[2], pa[3], bb[2], bb[3]);
            }
        }
    }
    cp_async_wait<0>();

    // Sin suma entre warps: cada warp vio todas las llaves para sus 16 filas.
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l8 += __shfl_xor_sync(0xffffffffu, l8, 1);
    l8 += __shfl_xor_sync(0xffffffffu, l8, 2);
    const int r0 = rg * 16 + g;   // fila del pedazo (pieza 2rg, cabeza g) y r0 + 8 (pieza 2rg + 1)
    const int64_t base = (static_cast<int64_t>(b * 2 + h) * n_pedazos + s) * bloque_m;
#pragma unroll
    for (int nd = 0; nd < 8; ++nd) {
        *reinterpret_cast<float2*>(&ap[(base + r0) * 64 + nd * 8 + tig * 2]) = make_float2(o[nd][0], o[nd][1]);
        *reinterpret_cast<float2*>(&ap[(base + r0 + 8) * 64 + nd * 8 + tig * 2]) = make_float2(o[nd][2], o[nd][3]);
    }
    if (tig == 0) {
        lp[base + r0] = l0;
        lp[base + r0 + 8] = l8;


    }
}

__device__ __forceinline__ float e4m3_a_float(uint8_t x) {
    const int exp = (x >> 3) & 15, man = x & 7;
    float y = exp ? ldexpf(1.f + man * (1.f / 8.f), exp - 7)
                  : ldexpf(man * (1.f / 8.f), -6);
    return (x & 128) ? -y : y;
}

__device__ __forceinline__ float reducir_warp(float x) {
    for (int delta = 16; delta > 0; delta >>= 1)
        x += __shfl_down_sync(0xffffffffu, x, delta);
    return __shfl_sync(0xffffffffu, x, 0);
}

// Un warp/CTA por (peticion, pieza nueva, cabeza Q) para combinar los
// splits y realizar la actualizacion causal de los 8 nuevos tokens.
// No se necesitan atomicos, locks ni barreras entre CTAs.
__global__ void __launch_bounds__(32, 8)
cerrar_verificar_museone(const __nv_bfloat16* __restrict__ qkv, int64_t paso_qkv,
                         const int* __restrict__ seq_lens, const int64_t* __restrict__ cache_loc,
                         uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                         const float* __restrict__ ks_ptr, const float* __restrict__ vs_ptr,
                         const float* __restrict__ lp, const float* __restrict__ ap,
                         __nv_bfloat16* __restrict__ out, int chunk, int n_pedazos,
                         float qk_escala, float escala_log2, float c_ref) {
    const int pid = blockIdx.x;
    const int cabeza = pid % 16;
    const int tok = pid / 16;
    const int j = tok % 8;
    const int b = tok / 8;
    const int h = cabeza / 8, g = cabeza % 8;
    const int lane = threadIdx.x;
    const int d0 = lane * 2;
    const int d1 = d0 + 1;
    const int seq = seq_lens[b];
    const int validos = min(n_pedazos, (seq + chunk - 1) / chunk);
    const int bh = b * 2 + h;
    const int r = j * 8 + g;
    const __nv_bfloat16* token = qkv + static_cast<int64_t>(tok) * paso_qkv;
    float acc0 = 0.f, acc1 = 0.f, suma = 0.f;
    for (int s = 0; s < validos; ++s) {
        const int64_t idx = (static_cast<int64_t>(bh) * n_pedazos + s) * 64 + r;
        suma += lp[idx];
        acc0 += ap[idx * 64 + d0];
        acc1 += ap[idx * 64 + d1];
    }
    const float ks = ks_ptr[0], vs = vs_ptr[0];
    const float q0 = __bfloat162float(token[cabeza * 64 + d0]);
    const float q1 = __bfloat162float(token[cabeza * 64 + d1]);
    const float nq = reducir_warp(q0 * q0 + q1 * q1);
    const float qinv = rsqrtf(nq / 64.f + 1e-5f) * qk_escala;
    const float qn0 = __bfloat162float(__float2bfloat16_rn(q0 * qinv));
    const float qn1 = __bfloat162float(__float2bfloat16_rn(q1 * qinv));

    // Solo la CTA (j=i, g=0) de cada cabeza KV escribe el token i.
    // Las otras CTAs calculan desde QKV sin leer filas escritas concurrentemente.
    const bool cabeza_escritora = (g == 0);
    for (int i = 0; i <= j; ++i) {
        const __nv_bfloat16* nuevo = qkv + static_cast<int64_t>(b * 8 + i) * paso_qkv;
        const float k0 = __bfloat162float(nuevo[(16 + h) * 64 + d0]);
        const float k1 = __bfloat162float(nuevo[(16 + h) * 64 + d1]);
        const float nk = reducir_warp(k0 * k0 + k1 * k1);
        const float kinv = rsqrtf(nk / 64.f + 1e-5f) / ks;
        const uint8_t k80 = static_cast<uint8_t>(__nv_cvt_float_to_fp8(k0 * kinv, __NV_SATFINITE, __NV_E4M3));
        const uint8_t k81 = static_cast<uint8_t>(__nv_cvt_float_to_fp8(k1 * kinv, __NV_SATFINITE, __NV_E4M3));
        const float v0 = __bfloat162float(nuevo[(18 + h) * 64 + d0]);
        const float v1 = __bfloat162float(nuevo[(18 + h) * 64 + d1]);
        const uint8_t v80 = static_cast<uint8_t>(__nv_cvt_float_to_fp8(v0 / vs, __NV_SATFINITE, __NV_E4M3));
        const uint8_t v81 = static_cast<uint8_t>(__nv_cvt_float_to_fp8(v1 / vs, __NV_SATFINITE, __NV_E4M3));
        if (cabeza_escritora && j == i) {
            const int64_t base = cache_loc[b * 8 + i] * 128 + h * 64;
            kc[base + d0] = k80; kc[base + d1] = k81;
            vc[base + d0] = v80; vc[base + d1] = v81;
        }
        const float dot = reducir_warp(qn0 * e4m3_a_float(k80) + qn1 * e4m3_a_float(k81));
        const float p = exp2f(fminf(dot * ks * escala_log2 - c_ref, 100.f));
        suma += p;
        acc0 += p * e4m3_a_float(v80);
        acc1 += p * e4m3_a_float(v81);
    }
    const float a0 = acc0 / suma * vs, a1 = acc1 / suma * vs;
    const float nout = reducir_warp(a0 * a0 + a1 * a1);
    const float inv = rsqrtf(nout / 64.f + 1e-5f);
    const float gate0 = __bfloat162float(token[(20 + cabeza) * 64 + d0]);
    const float gate1 = __bfloat162float(token[(20 + cabeza) * 64 + d1]);
    out[static_cast<int64_t>(tok) * (16 * 64) + cabeza * 64 + d0] =
        __float2bfloat16_rn(a0 * inv / (1.f + expf(-gate0)));
    out[static_cast<int64_t>(tok) * (16 * 64) + cabeza * 64 + d1] =
        __float2bfloat16_rn(a1 * inv / (1.f + expf(-gate1)));
}
}  // namespace muse

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <mutex>
#include <array>

void museone_parcial(torch::Tensor qkv, torch::Tensor seq, torch::Tensor tabla,
                     torch::Tensor kc, torch::Tensor vc, torch::Tensor ks,
                     torch::Tensor lp, torch::Tensor ap, int64_t pagina, int64_t chunk,
                     int64_t n_pedazos, int64_t bloque_m,
                     double qk_escala, double escala_log2, double c_ref) {
    const c10::cuda::CUDAGuard guard(qkv.device());
    const int dev = qkv.get_device();
    const int mem = muse::ETAPAS * muse::BLOQUE_ETAPA + 8 * 16 * 128;
    static std::array<std::once_flag, 64> flags;
    TORCH_CHECK(dev >= 0 && dev < 64, "device fuera del rango admitido");
    if (pagina == 1) {
        std::call_once(flags[dev], [=]() {
            C10_CUDA_CHECK(cudaFuncSetAttribute(muse::parcial_verificar_museone<1>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, mem));
        });
        muse::parcial_verificar_museone<1><<<dim3(n_pedazos, seq.size(0)), muse::HILOS, mem,
                                                  at::cuda::getCurrentCUDAStream(dev)>>>(
            reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), qkv.stride(0), seq.data_ptr<int>(),
            tabla.data_ptr<int>(), tabla.stride(0), 1,
            reinterpret_cast<const uint8_t*>(kc.data_ptr()), reinterpret_cast<const uint8_t*>(vc.data_ptr()),
            ks.data_ptr<float>(), qk_escala, escala_log2, c_ref,
            lp.data_ptr<float>(), ap.data_ptr<float>(), chunk, n_pedazos, bloque_m);
    } else {
        TORCH_CHECK(pagina == 64, "MuseOne CUDA: solo page_size 1 o 64");
        // La configuracion inicial de smem debe realizarse para CADA especializacion.
        static std::array<std::once_flag, 64> flags64;
        std::call_once(flags64[dev], [=]() {
            C10_CUDA_CHECK(cudaFuncSetAttribute(muse::parcial_verificar_museone<64>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, mem));
        });
        muse::parcial_verificar_museone<64><<<dim3(n_pedazos, seq.size(0)), muse::HILOS, mem,
                                                   at::cuda::getCurrentCUDAStream(dev)>>>(
            reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), qkv.stride(0), seq.data_ptr<int>(),
            tabla.data_ptr<int>(), tabla.stride(0), 64,
            reinterpret_cast<const uint8_t*>(kc.data_ptr()), reinterpret_cast<const uint8_t*>(vc.data_ptr()),
            ks.data_ptr<float>(), qk_escala, escala_log2, c_ref,
            lp.data_ptr<float>(), ap.data_ptr<float>(), chunk, n_pedazos, bloque_m);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void museone_cerrar(torch::Tensor qkv, torch::Tensor seq, torch::Tensor loc,
                    torch::Tensor kc, torch::Tensor vc, torch::Tensor ks,
                    torch::Tensor vs, torch::Tensor lp, torch::Tensor ap,
                    torch::Tensor out, int64_t chunk, int64_t n_pedazos,
                    double qk_escala, double escala_log2, double c_ref) {
    const c10::cuda::CUDAGuard guard(qkv.device());
    const int dev = qkv.get_device();
    muse::cerrar_verificar_museone<<<qkv.size(0) * 16, 32, 0, at::cuda::getCurrentCUDAStream(dev)>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), qkv.stride(0),
        seq.data_ptr<int>(), loc.data_ptr<int64_t>(),
        reinterpret_cast<uint8_t*>(kc.data_ptr()), reinterpret_cast<uint8_t*>(vc.data_ptr()),
        ks.data_ptr<float>(), vs.data_ptr<float>(), lp.data_ptr<float>(), ap.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), chunk, n_pedazos,
        qk_escala, escala_log2, c_ref);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _cargar_museone_cuda():
    if "modulo" not in _MUSEONE:
        try:
            from torch.utils.cpp_extension import load_inline
            declaraciones = """
void museone_parcial(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                     torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t,
                     int64_t, int64_t, double, double, double);
void museone_cerrar(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor, torch::Tensor, int64_t, int64_t, double, double, double);
"""
            _MUSEONE["modulo"] = load_inline(
                name="museone_cuda_verify_warp1",
                cpp_sources=declaraciones,
                cuda_sources=_CUDA_FUENTE + _CUDA_ANFITRION + _MUSEONE_FUENTE,
                functions=["museone_parcial", "museone_cerrar"],
                extra_cflags=["-O3", "-std=c++20"],
                extra_cuda_cflags=["-O3", "-std=c++20"],
                verbose=False,
            )
            print("MuseOne: verify/full CUDA warp-local y cierre nativo compilados", flush=True)
        except Exception as exc:
            print("MuseOne: fallo compilacion CUDA (FA3 como respaldo): " +
                  str(type(exc).__name__) + ": " + str(exc)[:1200], flush=True)
            _MUSEONE["modulo"] = None
    return _MUSEONE["modulo"]


def _metodo_museone_verificar(data):
    cfg = data["config"]
    batch, largo = cfg["batch"], cfg["q_len"]
    hq, hkv, dim, pagina = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    assert (largo, hq, hkv, dim) == (8, 16, 2, 64) and pagina in (1, 64)
    cota = float(cfg["qk_scale_factor"]) * dim * float(cfg["sm_scale"]) * LOG2E
    if not 0.0 < cota <= 75.0:
        return None
    modulo = _cargar_museone_cuda()
    if modulo is None:
        return None
    tabla = data["page_table"]
    max_llaves = max(1, tabla.shape[1] * pagina)
    bloque_n = 64 if max_llaves <= 4096 else 128
    n_bh = batch * hkv
    n_pedazos = max(1, min(triton.cdiv(PROGRAMAS_POR_SM * _n_sms(data["qkv"].device), n_bh),
                           triton.cdiv(max_llaves, 2 * bloque_n)))
    chunk = triton.cdiv(triton.cdiv(max_llaves, n_pedazos), bloque_n) * bloque_n
    n_pedazos = max(1, triton.cdiv(max_llaves, chunk))
    qkv = data["qkv"].reshape(batch * largo, -1)
    lp = torch.empty((n_bh * n_pedazos * 64,), dtype=torch.float32, device=qkv.device)
    ap = torch.empty((n_bh * n_pedazos * 64, 64), dtype=torch.float32, device=qkv.device)
    out = torch.empty((batch, largo, hq * 64), dtype=torch.bfloat16, device=qkv.device)
    esc_log2 = float(cfg["sm_scale"]) * LOG2E
    c_ref = cota - HOLGURA
    modulo.museone_parcial(
        qkv, data["seq_lens"], tabla, data["k_cache"], data["v_cache"],
        data["k_scale"], lp, ap, pagina, chunk, n_pedazos, 64,
        float(cfg["qk_scale_factor"]), esc_log2, c_ref)
    modulo.museone_cerrar(
        qkv, data["seq_lens"], data["cache_loc"], data["k_cache"],
        data["v_cache"], data["k_scale"], data["v_scale"],
        lp, ap, out, chunk, n_pedazos, float(cfg["qk_scale_factor"]), esc_log2, c_ref)
    return out


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    # Solo afecta a verify/full grande: decode y ventana permanecen byte a byte como v17.
    nativo = (cfg["q_len"] == 8 and cfg["window"] == 0
              and not cfg["rope_theta"] and cfg["head_dim"] == 64
              and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2
              and cfg["page_size"] in (1, 64) and cfg["batch"] >= 16
              and data["qkv"].device.type == "cuda")
    if nativo:
        resultado = _metodo_museone_verificar(data)
        if resultado is not None:
            return resultado
    verify_completo = (cfg["q_len"] == 8 and cfg["window"] == 0 and not cfg["rope_theta"]
                       and cfg["head_dim"] == 64 and cfg["num_q_heads"] % cfg["num_kv_heads"] == 0
                       and data["qkv"].device.type == "cuda")
    if verify_completo:
        api = _cargar_fa3_v17()
        if api is not None:
            return _metodo_fa3_v17(data, api)
    return _kernel_v16(data)
