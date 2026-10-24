# okc: problem=muse_attention_decode
"""Atencion con puerta de Muse Spark Open (un paso de decodificacion o un bloque de verificacion de 8 piezas) en dos
kernels de Triton, en vez de los cinco de SGLang (FusedQKNorm, store_cache, q a FP8, FA3 y rms_norm_gate).

Inspirado en ASRN (github.com/KIROKOWA/ASRN): como su atencion por casilleros fusionada, los puntajes nunca se escriben
en la memoria de la GPU, y los pedazos del contexto se juntan con la misma composicion asociativa que usa su escaneo
paralelo: (m1, l1, a1) y (m2, l2, a2) dan (m, l1 2^(m1-m) + l2 2^(m2-m), a1 2^(m1-m) + a2 2^(m2-m)). Las filas sin
llaves validas llevan un maximo finito muy negativo, como _softmax_valido de ASRN, para que nunca den NaN.

  _parcial  un programa por (pedazo del contexto, peticion, par de cabezas kv): normaliza (y rota) sus consultas, lee
            las llaves y valores FP8 ya guardados del pedazo y deja (maximo, suma, acumulado) del softmax en linea.
  _final    una por (peticion, cabeza kv, fila de consulta): junta los pedazos, agrega las L llaves nuevas del bloque
            (normalizadas, rotadas, redondeadas a FP8, que tambien guarda en el cache) y aplica rmsnorm x sigmoid(gate).

_parcial solo lee posiciones anteriores al bloque y _final solo escribe las del bloque, asi que no hay carrera entre
leer y escribir el cache. Sin sincronizar con la CPU y sin ramas sobre valores: se puede capturar en un CUDA graph.

v4 agrega:
  Planificador (teoria de colas): los trabajos no se reparten por la peticion mas larga posible sino por las longitudes
    reales, que cada programa lee en la GPU: el total de llaves se parte en trabajos del mismo tamano, tantos como
    lugares tiene la maquina (SMs x programas que caben por SM x 2 tandas). Sin programas vacios ni cola al final.
  Organizador de memoria: doble bufer en registros (las llaves y valores del bloque siguiente, y la tabla de paginas
    del que sigue a ese, ya van en camino mientras se calcula este), lecturas del cache marcadas evict_first (se leen
    una sola vez: que no saquen del L2 lo que si se reusa) y una configuracion por forma elegida para que quepan mas
    programas por SM sin derramar registros (medida con ptxas para sm_90)."""

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


RMS_EPS = 1e-5
LOG2E = 1.4426950408889634
VACIO = tl.constexpr(-1e30)   # maximo de una fila sin llaves validas: finito, para que 2^(m - m') no de NaN
ROPE_POSITIONS = 1 << 21      # como el RotaryEmbedding de SGLang (y baseline.py)
TANDAS = 2                    # trabajos de _parcial: 2 tandas de los programas que caben a la vez en la GPU
BLOQUE_S = 128                # _final junta los pedazos de a tantos
# Por forma (q_len): (llaves por vuelta, cabezas kv por programa, warps, programas que caben por SM), elegida con
# compilar_sm90.py (ptxas -v para sm_90) por la mayor cantidad de bytes en vuelo por SM sin derramar registros:
#   decode: 213 registros -> 2 por SM, 64 KB en vuelo, filas del cache completas (las dos cabezas kv a la vez)
#   verify: 166 registros -> 3 por SM, 48 KB en vuelo (con dos cabezas derrama registros; la v3 dejaba 1 por SM)
CONFIG = {1: (64, 2, 4, 2), 8: (64, 1, 4, 3)}
_ROPE = {}
_SMS = {}


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
def _paso(q, k, v, valido, m, suma, acc, escala, EN_CPU: tl.constexpr):
    """Una vuelta del softmax en linea: las llaves k y valores v ([N, 64] bf16) sobre las consultas q ([M, 64] bf16)."""
    if EN_CPU:  # el interprete de Triton multiplica mal en bf16: alli, en float32 exacto
        sc = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32)), input_precision="ieee") * escala
    else:
        sc = tl.dot(q, tl.trans(k)) * escala
    sc = tl.where(valido, sc, float("-inf"))
    m_nuevo = tl.maximum(m, tl.max(sc, 1))
    alfa = tl.math.exp2(m - m_nuevo)
    p = tl.math.exp2(sc - m_nuevo[:, None])
    suma = suma * alfa + tl.sum(p, 1)
    if EN_CPU:
        pv = tl.dot(p.to(tl.bfloat16).to(tl.float32), v.to(tl.float32), input_precision="ieee")
    else:
        pv = tl.dot(p.to(tl.bfloat16), v)
    return m_nuevo, suma, acc * alfa[:, None] + pv


@triton.jit
def _guardar(MP, LP, AP, donde, fila_ok, m, suma, acc):
    d = tl.arange(0, 64)
    tl.store(MP + donde, m, mask=fila_ok)
    tl.store(LP + donde, suma, mask=fila_ok)
    tl.store(AP + donde[:, None] * 64 + d[None, :], acc, mask=fila_ok[:, None])


@triton.jit
def _reparto(SEQ, B: tl.constexpr, BB: tl.constexpr, VENTANA: tl.constexpr, n_trabajos, min_chunk,
             BLOQUE_N: tl.constexpr):
    """El plan de trabajos, igual en todos los programas: las llaves ya guardadas que ve cada peticion (lens), el
    tamano de trabajo (chunk: el total entre n_trabajos, en vueltas enteras, al menos min_chunk), cuantos trabajos
    le tocan a cada una (n) y donde terminan sus trabajos en la lista (acum, suma acumulada de n)."""
    c = tl.arange(0, BB)
    seqs = tl.load(SEQ + c, mask=c < B, other=0)
    if VENTANA > 0:
        lens = tl.minimum(seqs, VENTANA - 1)
    else:
        lens = seqs
    total = tl.sum(lens, 0)
    chunk = tl.maximum(((total + n_trabajos - 1) // n_trabajos + BLOQUE_N - 1) // BLOQUE_N * BLOQUE_N, min_chunk)
    n = (lens + chunk - 1) // chunk
    return c, seqs, lens, chunk, n, tl.cumsum(n, 0)


@triton.jit
def _parcial(QKV, SEQ, TABLA, KC, VC, KS, ROPE, MP, LP, AP, paso_qkv, paso_tabla, n_trabajos, n_lista, min_chunk,
             qk_escala, escala_log2, B: tl.constexpr, BB: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
             L: tl.constexpr, G: tl.constexpr, PAGINA: tl.constexpr, VENTANA: tl.constexpr, CON_ROPE: tl.constexpr,
             BLOQUE_M: tl.constexpr, BLOQUE_N: tl.constexpr, PAR: tl.constexpr, EN_CPU: tl.constexpr):
    """Un trabajo de la lista (program_id 0) para PAR cabezas kv seguidas (program_id 1)."""
    w = tl.program_id(0)
    bp = tl.program_id(1)
    h0 = bp * PAR
    c, seqs, lens, chunk, n, acum = _reparto(SEQ, B, BB, VENTANA, n_trabajos, min_chunk, BLOQUE_N)
    b = tl.sum((acum <= w).to(tl.int32), 0)  # la peticion de este trabajo; B si sobra
    hay = b < B
    b = tl.minimum(b, B - 1)
    es_b = c == b
    seq = tl.sum(tl.where(es_b, seqs, 0), 0)
    s = w - tl.sum(tl.where(es_b, acum - n, 0), 0)  # cual de los trabajos de su peticion
    inicio = seq - tl.sum(tl.where(es_b, lens, 0), 0) + s * chunk
    fin = tl.where(hay, tl.minimum(inicio + chunk, seq), inicio)

    filas = tl.arange(0, BLOQUE_M)
    fila_ok = filas < L * G
    j = tl.where(fila_ok, filas // G, 0)
    g = filas % G
    pos = seq + j
    d = tl.arange(0, 64)
    base_q = QKV + (b * L + j).to(tl.int64) * paso_qkv + (h0 * G + g) * 64
    q0 = _norma_rope(base_q, pos, ROPE, qk_escala, CON_ROPE).to(tl.bfloat16)  # filas de relleno: copias de la fila 0
    if PAR == 2:
        q1 = _norma_rope(base_q + G * 64, pos, ROPE, qk_escala, CON_ROPE).to(tl.bfloat16)
    escala = tl.load(KS) * escala_log2

    m0 = tl.full([BLOQUE_M], VACIO, tl.float32)
    suma0 = tl.zeros([BLOQUE_M], tl.float32)
    acc0 = tl.zeros([BLOQUE_M, 64], tl.float32)
    m1, suma1, acc1 = m0, suma0, acc0

    # Doble bufer: antes de la vuelta t0 ya estan pedidas sus llaves y valores (k0, v0, k1, v1) y las filas del
    # cache de la vuelta siguiente (fila_sig). En la vuelta se piden las llaves de la siguiente y la tabla de la
    # que sigue a esa, y despues se calcula con lo que ya llego.
    col = h0 * 64 + d[None, :]
    t = inicio + tl.arange(0, BLOQUE_N)
    ok = t < fin
    off = _filas(TABLA, b, paso_tabla, t, fin, PAGINA)[:, None] * (HKV * 64) + col
    k0 = tl.load(KC + off, mask=ok[:, None], other=0.0, eviction_policy="evict_first")
    v0 = tl.load(VC + off, mask=ok[:, None], other=0.0, eviction_policy="evict_first")
    k1, v1 = k0, v0
    if PAR == 2:
        k1 = tl.load(KC + off + 64, mask=ok[:, None], other=0.0, eviction_policy="evict_first")
        v1 = tl.load(VC + off + 64, mask=ok[:, None], other=0.0, eviction_policy="evict_first")
    fila_sig = _filas(TABLA, b, paso_tabla, t + BLOQUE_N, fin, PAGINA)
    for t0 in range(inicio, fin, BLOQUE_N):
        t = t0 + tl.arange(0, BLOQUE_N)
        ok = t < fin
        valido = ok[None, :]
        if VENTANA > 0:
            valido = valido & (t[None, :] > pos[:, None] - VENTANA)
        ok_sig = t + BLOQUE_N < fin
        off = fila_sig[:, None] * (HKV * 64) + col
        k0_sig = tl.load(KC + off, mask=ok_sig[:, None], other=0.0, eviction_policy="evict_first")
        v0_sig = tl.load(VC + off, mask=ok_sig[:, None], other=0.0, eviction_policy="evict_first")
        k1_sig, v1_sig = k0_sig, v0_sig
        if PAR == 2:
            k1_sig = tl.load(KC + off + 64, mask=ok_sig[:, None], other=0.0, eviction_policy="evict_first")
            v1_sig = tl.load(VC + off + 64, mask=ok_sig[:, None], other=0.0, eviction_policy="evict_first")
        fila_sig = _filas(TABLA, b, paso_tabla, t + 2 * BLOQUE_N, fin, PAGINA)
        m0, suma0, acc0 = _paso(q0, k0.to(tl.bfloat16), v0.to(tl.bfloat16), valido, m0, suma0, acc0, escala, EN_CPU)
        if PAR == 2:
            m1, suma1, acc1 = _paso(q1, k1.to(tl.bfloat16), v1.to(tl.bfloat16), valido, m1, suma1, acc1, escala,
                                    EN_CPU)
        k0, v0, k1, v1 = k0_sig, v0_sig, k1_sig, v1_sig

    guarda = fila_ok & hay
    bh = b * HKV + h0
    _guardar(MP, LP, AP, (bh * n_lista + w) * BLOQUE_M + filas, guarda, m0, suma0, acc0)
    if PAR == 2:
        _guardar(MP, LP, AP, ((bh + 1) * n_lista + w) * BLOQUE_M + filas, guarda, m1, suma1, acc1)


@triton.jit
def _final(QKV, SEQ, LOC, KC, VC, KS, VS, ROPE, MP, LP, AP, OUT, paso_qkv, n_trabajos, n_lista, min_chunk, qk_escala,
           escala_log2, B: tl.constexpr, BB: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr,
           G: tl.constexpr, VENTANA: tl.constexpr, CON_ROPE: tl.constexpr, BLOQUE_M: tl.constexpr,
           BLOQUE_N: tl.constexpr, BLOQUE_S: tl.constexpr, BLOQUE_L: tl.constexpr):
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

    # Juntar los trabajos de esta peticion de a BLOQUE_S (composicion asociativa del softmax en linea).
    c, seqs, lens, chunk, n, acum = _reparto(SEQ, B, BB, VENTANA, n_trabajos, min_chunk, BLOQUE_N)
    n_b = tl.sum(tl.where(c == b, n, 0), 0)
    primero = tl.sum(tl.where(c == b, acum - n, 0), 0)
    m = tl.max(tl.full([BLOQUE_S], VACIO, tl.float32), 0)
    suma = tl.sum(tl.zeros([BLOQUE_S], tl.float32), 0)
    acc = tl.zeros([64], tl.float32)
    for s0 in range(0, n_b, BLOQUE_S):
        sidx = s0 + tl.arange(0, BLOQUE_S)
        hay = sidx < n_b
        donde = (bh * n_lista + primero + sidx) * BLOQUE_M + r
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

    # Planificador: la lista de trabajos se arma en la GPU con las longitudes reales (_reparto); aqui solo se fija,
    # por la forma, cuantos trabajos caben: TANDAS x SMs x programas por SM, mas uno por peticion de redondeo.
    bloque_n, par, warps, por_sm = CONFIG[largo] if largo in CONFIG else CONFIG[8]
    if hkv % par:
        par = 1
    n_trabajos = TANDAS * _n_sms(device) * por_sm // (hkv // par)
    n_lista = n_trabajos + batch
    min_chunk = bloque_n if ventana else 4 * bloque_n  # en el contexto completo, al menos 4 vueltas por trabajo
    max_trabajos = triton.cdiv(ventana - 1 if ventana else tabla.shape[1] * pagina, min_chunk) * batch
    bloque_s = min(BLOQUE_S, triton.next_power_of_2(min(n_lista, max_trabajos)))
    bloque_m = max(16, triton.next_power_of_2(largo * g))
    n_bh = batch * hkv
    bb = triton.next_power_of_2(batch)

    mp = torch.empty((n_bh * n_lista * bloque_m,), dtype=torch.float32, device=device)
    lp = torch.empty_like(mp)
    ap = torch.empty((n_bh * n_lista * bloque_m, 64), dtype=torch.float32, device=device)
    out = torch.empty((batch, largo, hq * 64), dtype=torch.bfloat16, device=device)
    k_cache, v_cache = data["k_cache"], data["v_cache"]

    _parcial[(n_lista, hkv // par)](
        qkv, data["seq_lens"], tabla, k_cache, v_cache, data["k_scale"], rope, mp, lp, ap,
        qkv.stride(0), tabla.stride(0), n_trabajos, n_lista, min_chunk, qk_escala, escala_log2,
        B=batch, BB=bb, HQ=hq, HKV=hkv, L=largo, G=g, PAGINA=pagina, VENTANA=ventana, CON_ROPE=con_rope,
        BLOQUE_M=bloque_m, BLOQUE_N=bloque_n, PAR=par, EN_CPU=device.type == "cpu", num_warps=warps, num_stages=1)
    _final[(n_bh * largo * g,)](
        qkv, data["seq_lens"], data["cache_loc"], k_cache, v_cache, data["k_scale"], data["v_scale"], rope,
        mp, lp, ap, out, qkv.stride(0), n_trabajos, n_lista, min_chunk, qk_escala, escala_log2,
        B=batch, BB=bb, HQ=hq, HKV=hkv, L=largo, G=g, VENTANA=ventana, CON_ROPE=con_rope, BLOQUE_M=bloque_m,
        BLOQUE_N=bloque_n, BLOQUE_S=bloque_s, BLOQUE_L=triton.next_power_of_2(largo),
        num_warps=8 if bloque_s > 32 else 4)
    return out
