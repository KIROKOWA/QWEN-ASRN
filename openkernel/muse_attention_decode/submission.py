# okc: problem=muse_attention_decode
"""Atencion con puerta de Muse Spark Open (un paso de decodificacion o un bloque de verificacion de 8 piezas) en dos
kernels de Triton, en vez de los cinco de SGLang (FusedQKNorm, store_cache, q a FP8, FA3 y rms_norm_gate).

Inspirado en ASRN (github.com/KIROKOWA/ASRN): como su atencion por casilleros fusionada, los puntajes nunca se escriben
en la memoria de la GPU, y los pedazos del contexto se juntan con la misma composicion asociativa que usa su escaneo
paralelo: (m1, l1, a1) y (m2, l2, a2) dan (m, l1 2^(m1-m) + l2 2^(m2-m), a1 2^(m1-m) + a2 2^(m2-m)). Las filas sin
llaves validas llevan un maximo finito muy negativo, como _softmax_valido de ASRN, para que nunca den NaN.

  _parcial  una pieza de programa por (pedazo del contexto, peticion, cabeza kv): normaliza (y rota) sus consultas, lee
            las llaves y valores FP8 ya guardados del pedazo y deja (maximo, suma, acumulado) del softmax en linea.
  _final    una por (peticion, cabeza kv, fila de consulta): junta los pedazos, agrega las L llaves nuevas del bloque
            (normalizadas, rotadas, redondeadas a FP8, que tambien guarda en el cache) y aplica rmsnorm x sigmoid(gate).

_parcial solo lee posiciones anteriores al bloque y _final solo escribe las del bloque, asi que no hay carrera entre
leer y escribir el cache. Sin sincronizar con la CPU y sin ramas sobre valores: se puede capturar en un CUDA graph."""

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
BLOQUE_N = 64                 # llaves por vuelta de _parcial
MAX_PEDAZOS = 128             # tope de pedazos por (peticion, cabeza kv): _final los junta de una vez
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
def _parcial(QKV, SEQ, TABLA, KC, VC, KS, ROPE, MP, LP, AP, paso_qkv, paso_tabla, chunk, n_pedazos, qk_escala,
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
    for t0 in range(inicio, fin, BLOQUE_N):
        t = t0 + tl.arange(0, BLOQUE_N)
        ok = t < fin
        if PAGINA == 1:
            fila_c = tl.load(TABLA + b * paso_tabla + t, mask=ok, other=0).to(tl.int64)
        else:
            fila_c = tl.load(TABLA + b * paso_tabla + t // PAGINA, mask=ok, other=0).to(tl.int64) * PAGINA + t % PAGINA
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
    tl.store(MP + donde, m)
    tl.store(LP + donde, suma)
    tl.store(AP + donde[:, None] * 64 + d[None, :], acc)


@triton.jit
def _final(QKV, SEQ, LOC, KC, VC, KS, VS, ROPE, MP, LP, AP, OUT, paso_qkv, n_pedazos, qk_escala, escala_log2,
           HQ: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr, G: tl.constexpr, CON_ROPE: tl.constexpr,
           BLOQUE_M: tl.constexpr, BLOQUE_S: tl.constexpr):
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

    # Juntar los pedazos de _parcial de una vez (composicion asociativa del softmax en linea).
    sidx = tl.arange(0, BLOQUE_S)
    hay = sidx < n_pedazos
    donde = (bh * n_pedazos + sidx) * BLOQUE_M + r
    ms = tl.load(MP + donde, mask=hay, other=VACIO)
    m = tl.max(ms, 0)
    w = tl.math.exp2(ms - m) * hay
    suma = tl.sum(tl.load(LP + donde, mask=hay, other=0.0) * w, 0)
    acc = tl.sum(tl.load(AP + donde[:, None] * 64 + d[None, :], mask=hay[:, None], other=0.0) * w[:, None], 0)

    fila_q = QKV + (b * L + j).to(tl.int64) * paso_qkv + cabeza * 64 + uno
    q = tl.reshape(_norma_rope(fila_q, seq + j + uno, ROPE, qk_escala, CON_ROPE), (64,))
    q = q.to(tl.bfloat16).to(tl.float32)
    escribe = r == 0  # una fila por (peticion, cabeza kv) guarda las llaves y valores nuevos

    # Las L piezas del bloque: cada fila j ve las i <= j (siempre dentro de la ventana, que es mucho mas larga).
    for i in tl.static_range(L):
        tok = (b * L + i).to(tl.int64)
        fila_k = QKV + tok * paso_qkv + (HQ + h) * 64 + uno
        k = tl.reshape(_norma_rope(fila_k, seq + i + uno, ROPE, 1.0, CON_ROPE), (64,))
        k8 = _a_fp8(k / ks)
        v8 = _a_fp8(tl.load(QKV + tok * paso_qkv + (HQ + HKV + h) * 64 + d).to(tl.float32) / vs)
        destino = tl.load(LOC + tok) * (HKV * 64) + h * 64 + d
        tl.store(KC + destino, k8, mask=escribe & (d < 64))
        tl.store(VC + destino, v8, mask=escribe & (d < 64))
        sc = tl.sum(q * k8.to(tl.float32), 0) * ks * escala_log2
        sc = tl.where(i <= j, sc, float("-inf"))
        m_nuevo = tl.maximum(m, sc)
        alfa = tl.math.exp2(m - m_nuevo)
        p = tl.math.exp2(sc - m_nuevo)
        suma = suma * alfa + p
        acc = acc * alfa + p * v8.to(tl.float32)
        m = m_nuevo

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

    # Pedazos del contexto ya guardado, por la forma (no por los valores): unos dos programas por SM.
    max_llaves = ventana - 1 if ventana else tabla.shape[1] * pagina
    n_bh = batch * hkv
    n_pedazos = max(1, min(MAX_PEDAZOS, triton.cdiv(max_llaves, BLOQUE_N), triton.cdiv(2 * _n_sms(device), n_bh)))
    chunk = triton.cdiv(triton.cdiv(max_llaves, n_pedazos), BLOQUE_N) * BLOQUE_N
    n_pedazos = max(1, triton.cdiv(max_llaves, chunk))
    bloque_m = max(16, triton.next_power_of_2(largo * g))

    mp = torch.empty((n_bh * n_pedazos * bloque_m,), dtype=torch.float32, device=device)
    lp = torch.empty_like(mp)
    ap = torch.empty((n_bh * n_pedazos * bloque_m, 64), dtype=torch.float32, device=device)
    out = torch.empty((batch, largo, hq * 64), dtype=torch.bfloat16, device=device)
    k_cache, v_cache = data["k_cache"], data["v_cache"]

    _parcial[(n_pedazos, n_bh)](
        qkv, data["seq_lens"], tabla, k_cache, v_cache, data["k_scale"], rope, mp, lp, ap,
        qkv.stride(0), tabla.stride(0), chunk, n_pedazos, qk_escala, escala_log2,
        HQ=hq, HKV=hkv, L=largo, G=g, PAGINA=pagina, VENTANA=ventana, CON_ROPE=con_rope, BLOQUE_M=bloque_m,
        BLOQUE_N=BLOQUE_N, EN_CPU=device.type == "cpu", num_warps=4, num_stages=2)
    _final[(n_bh * largo * g,)](
        qkv, data["seq_lens"], data["cache_loc"], k_cache, v_cache, data["k_scale"], data["v_scale"], rope,
        mp, lp, ap, out, qkv.stride(0), n_pedazos, qk_escala, escala_log2,
        HQ=hq, HKV=hkv, L=largo, G=g, CON_ROPE=con_rope, BLOQUE_M=bloque_m,
        BLOQUE_S=triton.next_power_of_2(n_pedazos), num_warps=2)
    return out
