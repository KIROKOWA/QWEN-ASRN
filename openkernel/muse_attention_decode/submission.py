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
PROGRAMAS_POR_SM = 16         # _parcial: muchos programas en vuelo, para tapar la espera de las lecturas sueltas
BLOQUE_S = 128                # _final junta los pedazos de a tantos
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
def _parcial(QKV, SEQ, TABLA, KC, VC, KS, ROPE, MP, LP, AP, paso_qkv, paso_tabla, chunk, n_pedazos, qk_escala,
             escala_log2, HQ: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr, G: tl.constexpr,
             PAGINA: tl.constexpr, VENTANA: tl.constexpr, CON_ROPE: tl.constexpr, BLOQUE_M: tl.constexpr,
             BLOQUE_N: tl.constexpr, PAR: tl.constexpr, EN_CPU: tl.constexpr):
    """Un pedazo del contexto para PAR cabezas kv seguidas (h0, h0 + 1): con PAR = 2 cada programa pide las dos
    mitades de cada fila suelta del cache (128 bytes seguidos) a la vez, y no una mitad cada uno en otro momento."""
    s = tl.program_id(0)
    bp = tl.program_id(1)
    b = bp // (HKV // PAR)
    h0 = (bp % (HKV // PAR)) * PAR
    seq = tl.load(SEQ + b)
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

    # Las llaves ya guardadas que ve alguna fila del bloque: [lo, seq). La ventana de la fila j empieza en seq + j - V + 1.
    if VENTANA > 0:
        lo = tl.maximum(seq - VENTANA + 1, 0)
    else:
        lo = seq * 0
    inicio = lo + s * chunk
    fin = tl.minimum(inicio + chunk, seq)
    escala = tl.load(KS) * escala_log2

    m0 = tl.full([BLOQUE_M], VACIO, tl.float32)
    suma0 = tl.zeros([BLOQUE_M], tl.float32)
    acc0 = tl.zeros([BLOQUE_M, 64], tl.float32)
    m1, suma1, acc1 = m0, suma0, acc0
    # La tabla de paginas de la vuelta siguiente se pide antes de leer las llaves de esta: asi la lectura de la tabla
    # y la de las filas sueltas del cache que dependen de ella no se esperan una a la otra.
    fila_sig = _filas(TABLA, b, paso_tabla, inicio + tl.arange(0, BLOQUE_N), fin, PAGINA)
    for t0 in range(inicio, fin, BLOQUE_N):
        t = t0 + tl.arange(0, BLOQUE_N)
        ok = t < fin
        fila_c = fila_sig
        fila_sig = _filas(TABLA, b, paso_tabla, t + BLOQUE_N, fin, PAGINA)
        off = fila_c[:, None] * (HKV * 64) + h0 * 64 + d[None, :]
        k0 = tl.load(KC + off, mask=ok[:, None], other=0.0).to(tl.bfloat16)
        v0 = tl.load(VC + off, mask=ok[:, None], other=0.0).to(tl.bfloat16)
        if PAR == 2:
            k1 = tl.load(KC + off + 64, mask=ok[:, None], other=0.0).to(tl.bfloat16)
            v1 = tl.load(VC + off + 64, mask=ok[:, None], other=0.0).to(tl.bfloat16)
        valido = ok[None, :]
        if VENTANA > 0:
            valido = valido & (t[None, :] > pos[:, None] - VENTANA)
        m0, suma0, acc0 = _paso(q0, k0, v0, valido, m0, suma0, acc0, escala, EN_CPU)
        if PAR == 2:
            m1, suma1, acc1 = _paso(q1, k1, v1, valido, m1, suma1, acc1, escala, EN_CPU)

    bh = b * HKV + h0
    _guardar(MP, LP, AP, (bh * n_pedazos + s) * BLOQUE_M + filas, fila_ok, m0, suma0, acc0)
    if PAR == 2:
        _guardar(MP, LP, AP, ((bh + 1) * n_pedazos + s) * BLOQUE_M + filas, fila_ok, m1, suma1, acc1)


@triton.jit
def _final(QKV, SEQ, LOC, KC, VC, KS, VS, ROPE, MP, LP, AP, OUT, paso_qkv, n_pedazos, qk_escala, escala_log2,
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

    # Pedazos del contexto ya guardado, por la forma (no por los valores): unos PROGRAMAS_POR_SM programas por SM. En
    # la ventana (2,047 llaves) bastan pedazos de una vuelta de 64; en el contexto completo, de al menos 4 vueltas de
    # 128, para que _final no tenga que juntar cientos.
    max_llaves = ventana - 1 if ventana else tabla.shape[1] * pagina
    bloque_n, vueltas = (64, 1) if max_llaves <= 4096 else (128, 4)
    par = 2 if hkv % 2 == 0 else 1
    n_bh = batch * hkv
    n_pedazos = max(1, min(triton.cdiv(PROGRAMAS_POR_SM * _n_sms(device), n_bh // par),
                           triton.cdiv(max_llaves, vueltas * bloque_n)))
    chunk = triton.cdiv(triton.cdiv(max_llaves, n_pedazos), bloque_n) * bloque_n
    n_pedazos = max(1, triton.cdiv(max_llaves, chunk))
    bloque_m = max(16, triton.next_power_of_2(largo * g))

    mp = torch.empty((n_bh * n_pedazos * bloque_m,), dtype=torch.float32, device=device)
    lp = torch.empty_like(mp)
    ap = torch.empty((n_bh * n_pedazos * bloque_m, 64), dtype=torch.float32, device=device)
    out = torch.empty((batch, largo, hq * 64), dtype=torch.bfloat16, device=device)
    k_cache, v_cache = data["k_cache"], data["v_cache"]

    _parcial[(n_pedazos, n_bh // par)](
        qkv, data["seq_lens"], tabla, k_cache, v_cache, data["k_scale"], rope, mp, lp, ap,
        qkv.stride(0), tabla.stride(0), chunk, n_pedazos, qk_escala, escala_log2,
        HQ=hq, HKV=hkv, L=largo, G=g, PAGINA=pagina, VENTANA=ventana, CON_ROPE=con_rope, BLOQUE_M=bloque_m,
        BLOQUE_N=bloque_n, PAR=par, EN_CPU=device.type == "cpu", num_warps=8 if bloque_m > 16 else 4, num_stages=2)
    _final[(n_bh * largo * g,)](
        qkv, data["seq_lens"], data["cache_loc"], k_cache, v_cache, data["k_scale"], data["v_scale"], rope,
        mp, lp, ap, out, qkv.stride(0), n_pedazos, qk_escala, escala_log2,
        HQ=hq, HKV=hkv, L=largo, G=g, CON_ROPE=con_rope, BLOQUE_M=bloque_m,
        BLOQUE_S=min(BLOQUE_S, triton.next_power_of_2(n_pedazos)), BLOQUE_L=triton.next_power_of_2(largo),
        num_warps=8 if n_pedazos > 32 else 4)
    return out
