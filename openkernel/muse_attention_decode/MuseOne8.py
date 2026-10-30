# okc: problem=muse_attention_decode
"""MuseOne8: el kernel de SGLang (baseline.py) reconstruido, de 5 lanzamientos a 3.

SGLang hace, por paso: FusedQKNorm (Triton), store_cache, q.to(fp8), flash_attn_with_kvcache (FA3) y rms_norm_gate.
Aqui:
  1. _preparar (Triton, un programa por pieza): RMSNorm de q y k, RoPE intercalado en las capas con ventana, q a E4M3
     (redondeada antes a bf16, como la salida de FusedQKNorm) y k, v a E4M3 (entre las escalas del cache cuando no son
     1) escritas en sus filas del cache. Una sola pasada sobre qkv: reemplaza FusedQKNorm, store_cache y el cast de q.
  2. FA3 (sgl_kernel.flash_attn.flash_attn_with_kvcache) con los mismos argumentos que SGLang: tensor cores de Hopper
     (wgmma) y su propia division del contexto.
  3. _salida (Triton): RMSNorm de la salida de FA3 por cabeza, por sigmoid(gate), en bf16.
Sin sincronizar con la CPU ni ramificar sobre valores: se captura en un CUDA graph. La atencion la hace FA3 sobre
todas las llaves (ventana de 2048 o todo el contexto), como en SGLang.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


RMS_EPS = 1e-5
ROPE_POSITIONS = 1 << 21   # como el RotaryEmbedding de SGLang
FP8 = torch.float8_e4m3fn
_ROPE = {}
_FA3 = {}


def _cos_sin(theta, d, device):
    """[2^21, d] float32 cos | sin, como lo arma SGLang; depende solo de theta (no de las entradas)."""
    clave = (theta, d, str(device))
    tabla = _ROPE.get(clave)
    if tabla is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        tabla = _ROPE[clave] = torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()
    return tabla


def _fa3():
    if "fn" not in _FA3:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        _FA3["fn"] = flash_attn_with_kvcache
    return _FA3["fn"]


@triton.jit
def _norma_rope(x_par, x_impar, cs, CON_ROPE: tl.constexpr):
    """RMSNorm sin peso de filas [R, 64] dadas por sus mitades par/impar ([R, 32]) y RoPE intercalado (cs: fila de
    cos | sin de la posicion)."""
    inv = tl.rsqrt((tl.sum(x_par * x_par, 1) + tl.sum(x_impar * x_impar, 1)) / 64 + 1e-5)
    x_par = x_par * inv[:, None]
    x_impar = x_impar * inv[:, None]
    if CON_ROPE:
        mitad = tl.arange(0, 32)
        c = tl.load(cs + mitad)[None, :]
        s = tl.load(cs + 32 + mitad)[None, :]
        x_par, x_impar = x_par * c - x_impar * s, x_par * s + x_impar * c
    return x_par, x_impar


@triton.jit
def _preparar(QKV, POS, LOC, ROPE, KC, VC, KS, VS, Q8, paso_qkv, HQ: tl.constexpr, HKV: tl.constexpr,
              HQP: tl.constexpr, HKP: tl.constexpr, CON_ROPE: tl.constexpr, CON_ESCALA: tl.constexpr):
    """Una pieza del bloque (program_id): q normalizada (y rotada) a E4M3 en Q8 [T, HQ, 64]; k normalizada (y rotada)
    y v a E4M3 en su fila del cache."""
    t = tl.program_id(0)
    base = QKV + t.to(tl.int64) * paso_qkv
    mitad = tl.arange(0, 32)
    d = tl.arange(0, 64)
    cs = ROPE
    if CON_ROPE:
        cs = ROPE + tl.load(POS + t).to(tl.int64) * 64

    # q: HQ cabezas
    hq = tl.arange(0, HQP)
    mq = hq < HQ
    q_par = tl.load(base + hq[:, None] * 64 + 2 * mitad[None, :], mask=mq[:, None], other=0.0).to(tl.float32)
    q_imp = tl.load(base + hq[:, None] * 64 + 2 * mitad[None, :] + 1, mask=mq[:, None], other=0.0).to(tl.float32)
    q_par, q_imp = _norma_rope(q_par, q_imp, cs, CON_ROPE)
    q = tl.reshape(tl.join(q_par, q_imp), (HQP, 64))
    q8 = q.to(tl.bfloat16).to(tl.float32).to(tl.float8e4nv)   # FusedQKNorm entrega bf16; q.to(fp8) despues
    tl.store(Q8 + (t.to(tl.int64) * HQ + hq[:, None]) * 64 + d[None, :], q8, mask=mq[:, None])

    # k y v: HKV cabezas, a su fila del cache
    hk = tl.arange(0, HKP)
    mk = hk < HKV
    k_par = tl.load(base + (HQ + hk[:, None]) * 64 + 2 * mitad[None, :], mask=mk[:, None], other=0.0).to(tl.float32)
    k_imp = tl.load(base + (HQ + hk[:, None]) * 64 + 2 * mitad[None, :] + 1, mask=mk[:, None],
                    other=0.0).to(tl.float32)
    k_par, k_imp = _norma_rope(k_par, k_imp, cs, CON_ROPE)
    k = tl.reshape(tl.join(k_par, k_imp), (HKP, 64)).to(tl.bfloat16).to(tl.float32)
    v = tl.load(base + (HQ + HKV + hk[:, None]) * 64 + d[None, :], mask=mk[:, None], other=0.0).to(tl.float32)
    if CON_ESCALA:
        k = k / tl.load(KS)
        v = v / tl.load(VS)
    fila = tl.load(LOC + t).to(tl.int64) * (HKV * 64)
    destino = fila + hk[:, None] * 64 + d[None, :]
    tl.store(KC + destino, k.to(tl.float8e4nv), mask=mk[:, None])
    tl.store(VC + destino, v.to(tl.float8e4nv), mask=mk[:, None])


@triton.jit
def _salida(A, QKV, OUT, n_filas, paso_a_tok, paso_a_cab, paso_qkv, HQ: tl.constexpr, HKV: tl.constexpr,
            FILAS: tl.constexpr):
    """FILAS filas (pieza, cabeza) por programa: RMSNorm de la salida de FA3 por sigmoid(gate), en bf16."""
    f = tl.program_id(0) * FILAS + tl.arange(0, FILAS)
    ok = f < n_filas
    t = (f // HQ).to(tl.int64)
    cab = f % HQ
    d = tl.arange(0, 64)
    a = tl.load(A + (t * paso_a_tok + cab * paso_a_cab)[:, None] + d[None, :], mask=ok[:, None], other=0.0)
    a = a.to(tl.float32)
    a = a * tl.rsqrt(tl.sum(a * a, 1) / 64 + 1e-5)[:, None]
    g = tl.load(QKV + (t * paso_qkv + (HQ + 2 * HKV + cab) * 64)[:, None] + d[None, :], mask=ok[:, None], other=0.0)
    o = a * tl.sigmoid(g.to(tl.float32))
    tl.store(OUT + (t * (HQ * 64) + cab * 64)[:, None] + d[None, :], o.to(tl.bfloat16), mask=ok[:, None])


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, largo = cfg["batch"], cfg["q_len"]
    hq, hkv, dim = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    assert dim == 64
    piezas = batch * largo
    qkv = data["qkv"].reshape(piezas, -1)
    device = qkv.device
    con_rope = bool(cfg["rope_theta"])
    rope = _cos_sin(cfg["rope_theta"], dim, device) if con_rope else qkv
    escalado = cfg["kv_scale"] != 1.0
    k_cache, v_cache = data["k_cache"], data["v_cache"]

    q8 = torch.empty((piezas, hq, dim), dtype=FP8, device=device)
    _preparar[(piezas,)](
        qkv, data["positions"], data["cache_loc"], rope, k_cache, v_cache, data["k_scale"], data["v_scale"], q8,
        qkv.stride(0), HQ=hq, HKV=hkv, HQP=triton.next_power_of_2(hq), HKP=max(2, triton.next_power_of_2(hkv)),
        CON_ROPE=con_rope, CON_ESCALA=escalado, num_warps=4)

    ventana = cfg["window"]
    escalas = {}
    if escalado:
        escalas = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    a = _fa3()(
        q=q8, k_cache=k_cache, v_cache=v_cache, page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=largo,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
        window_size=(ventana - 1, 0) if ventana else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"], **escalas)
    if isinstance(a, (tuple, list)):
        a = a[0]
    a = a.reshape(piezas, hq, dim)

    out = torch.empty((batch, largo, hq * dim), dtype=torch.bfloat16, device=device)
    filas = piezas * hq
    por_programa = 16
    _salida[(triton.cdiv(filas, por_programa),)](
        a, qkv, out, filas, a.stride(0), a.stride(1), qkv.stride(0), HQ=hq, HKV=hkv, FILAS=por_programa, num_warps=4)
    return out
