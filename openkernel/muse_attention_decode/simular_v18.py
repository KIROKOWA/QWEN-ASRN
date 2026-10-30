"""Simula en numpy (sin GPU ni torch) el kernel CUDA de v18 al nivel de los registros de cada hilo: ldmatrix (con y sin
.trans), mma.sync m16n8k32 E4M3 y m16n8k16 f16 con sus fragmentos tal como los define el PTX ISA, el swizzle de la
memoria compartida, __byte_perm, el reparto en segmentos, las mascaras, el maximo perezoso, los atomicos y el cierre.
Compara contra una referencia en float64 hecha aqui (la misma cuenta que reference.py) para atrapar errores de
indices de fragmentos antes de gastar un `okc check`.

    python simular_v18.py            # varios casos chicos: decode y verify, ventana y contexto completo, paginas 1 y 64
"""
import math
import sys

import numpy as np

TILE, ROWB = 64, 128
TAU, NEG = 8.0, -1e30
EPS = 1e-5
WINDOW = 2048
THETA = 500000.0
QK = 5.4730064863875
SM = 64 ** -0.5
LOG2E = 1.4426950408889634
HOLGURA = 40.0


# ---- formatos ---------------------------------------------------------------------------------------------------------
def _e4m3_tabla():
    vals = np.zeros(256)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        if e == 15 and m == 7:
            v = np.nan
        elif e == 0:
            v = m / 8 * 2.0 ** -6
        else:
            v = (1 + m / 8) * 2.0 ** (e - 7)
        vals[b] = -v if s else v
    return vals


E4M3 = _e4m3_tabla()
_POS = np.array([E4M3[b] for b in range(127)])  # 0 .. 448, crecientes


def a_e4m3(x):
    """float -> byte E4M3, redondeo al par mas cercano, saturando (satfinite)."""
    x = np.asarray(x, dtype=np.float64)
    s = np.signbit(x)
    a = np.minimum(np.abs(x), 448.0)
    i = np.searchsorted(_POS, a)          # _POS[i-1] < a <= _POS[i]
    i = np.clip(i, 1, 126)
    lo, hi = _POS[i - 1], _POS[i]
    elige_hi = (a - lo > hi - a) | ((a - lo == hi - a) & ((i % 2) == 0))
    k = np.where(a <= _POS[0], 0, np.where(elige_hi, i, i - 1))
    return (k | (s.astype(np.int64) << 7)).astype(np.uint8)


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def f16(x):
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


# ---- registros: un warp = 32 carriles, cada registro un uint32 ---------------------------------------------------------
def u8x4(b0, b1, b2, b3):
    return (np.uint32(b0) | (np.uint32(b1) << 8) | (np.uint32(b2) << 16) | (np.uint32(b3) << 24)).astype(np.uint32)


def bytes_de(r):
    r = np.asarray(r, dtype=np.uint32)
    return [((r >> (8 * i)) & 0xFF).astype(np.uint8) for i in range(4)]


def byte_perm(x, sel):
    b = bytes_de(x) + [np.zeros_like(bytes_de(x)[0])] * 4
    return u8x4(*[b[(sel >> (4 * i)) & 7] for i in range(4)])


def f16x2(lo, hi):
    l = np.asarray(lo, dtype=np.float32).astype(np.float16).view(np.uint16).astype(np.uint32)
    h = np.asarray(hi, dtype=np.float32).astype(np.float16).view(np.uint16).astype(np.uint32)
    return (l | (h << 16)).astype(np.uint32)


def de_f16x2(r):
    r = np.asarray(r, dtype=np.uint32)
    return ((r & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32),
            (r >> 16).astype(np.uint16).view(np.float16).astype(np.float32))


def e4m3x2_f16x2(x):
    b = bytes_de(x)
    return f16x2(E4M3[b[0]], E4M3[b[1]])


def ldsm_x4(smem, addr, trans=False):
    """addr[32]: la direccion de fila que da cada carril (matriz carril // 8, fila carril % 8). Devuelve 4 registros
    [32]. Sin .trans el carril T recibe (fila T/4, b16 2(T%4), 2(T%4)+1); con .trans (filas 2(T%4), 2(T%4)+1, col T/4)."""
    out = []
    for m in range(4):
        filas = [smem[addr[m * 8 + r]: addr[m * 8 + r] + 16] for r in range(8)]
        reg = np.zeros(32, dtype=np.uint32)
        for t in range(32):
            g, tig = t >> 2, t & 3
            if not trans:
                fb = filas[g][4 * tig: 4 * tig + 4]
                reg[t] = u8x4(*fb)
            else:
                a = filas[2 * tig][2 * g: 2 * g + 2]
                b = filas[2 * tig + 1][2 * g: 2 * g + 2]
                reg[t] = u8x4(a[0], a[1], b[0], b[1])
        out.append(reg)
    return out


def mma_fp8(c, a, b):
    """m16n8k32 row.col f32.e4m3.e4m3.f32: a[4][32], b[2][32], c[32][4] (in place)."""
    A = np.zeros((16, 32))
    B = np.zeros((32, 8))
    for t in range(32):
        g, tig = t >> 2, t & 3
        for i, (r, k0) in enumerate([(g, 0), (g + 8, 0), (g, 16), (g + 8, 16)]):
            A[r, k0 + 4 * tig: k0 + 4 * tig + 4] = E4M3[bytes_de(a[i][t:t + 1])].ravel()
        for i, k0 in enumerate([0, 16]):
            B[k0 + 4 * tig: k0 + 4 * tig + 4, g] = E4M3[bytes_de(b[i][t:t + 1])].ravel()
    _acumular(c, A @ B)


def mma_f16(c, a, b):
    """m16n8k16 row.col f32.f16.f16.f32."""
    A = np.zeros((16, 16))
    B = np.zeros((16, 8))
    for t in range(32):
        g, tig = t >> 2, t & 3
        for i, (r, k0) in enumerate([(g, 0), (g + 8, 0), (g, 8), (g + 8, 8)]):
            lo, hi = de_f16x2(a[i][t])
            A[r, k0 + 2 * tig], A[r, k0 + 2 * tig + 1] = lo, hi
        for i, k0 in enumerate([0, 8]):
            lo, hi = de_f16x2(b[i][t])
            B[k0 + 2 * tig, g], B[k0 + 2 * tig + 1, g] = lo, hi
    _acumular(c, A @ B)


def _acumular(c, D):
    for t in range(32):
        g, tig = t >> 2, t & 3
        c[t, 0] += D[g, 2 * tig]
        c[t, 1] += D[g, 2 * tig + 1]
        c[t, 2] += D[g + 8, 2 * tig]
        c[t, 3] += D[g + 8, 2 * tig + 1]


def swz(row, chunk):
    return row * ROWB + ((chunk ^ (row & 7)) << 4)


def shfl_max(x):  # xor 1, 2 dentro de los grupos de 4 carriles
    x = x.copy()
    for o in (1, 2):
        x = np.maximum(x, x[np.arange(32) ^ o])
    return x


# ---- el kernel ---------------------------------------------------------------------------------------------------------
class Warp:
    def __init__(self):
        self.o = np.zeros((8, 32, 4))
        self.l = np.zeros((32, 4))
        self.m0 = np.full(32, NEG)
        self.m8 = np.full(32, NEG)


def attend_tile(L, exp16, a, qa, ks, vs, t0, hi, seq, window, scl, n16, warp):
    lane = np.arange(32)
    h = warp >> 2
    g, tig, mi, ri = lane >> 2, lane & 3, lane >> 3, lane & 7
    kb_first = (warp & 3) if L == 1 else 0
    kb_end = (warp & 3) + 1 if L == 1 else n16
    if kb_first >= n16:
        return
    jg = 0 if L == 1 else 2 * (warp & 3)
    j8 = jg + 1
    edge = (t0 + TILE > hi) or (window > 0 and t0 <= seq + L - 1 - window) or (t0 + TILE > seq)
    for kb in range(kb_first, kb_end):
        kf = [ldsm_x4(ks, swz(kb * 16 + nb * 8 + ri, h * 4 + mi)) for nb in range(2)]
        s = np.zeros((2, 32, 4))
        for nb in range(2):
            mma_fp8(s[nb], qa[0], kf[nb][0:2])
            mma_fp8(s[nb], qa[1], kf[nb][2:4])
        s = (s.astype(np.float32) * np.float32(scl)).astype(np.float64)
        if edge:
            for nb in range(2):
                for i in range(4):
                    t = t0 + kb * 16 + nb * 8 + tig * 2 + (i & 1)
                    pos = seq + (jg if i < 2 else j8)
                    ok = (t < hi) & (t <= pos) & ((window == 0) | (t > pos - window))
                    s[nb][:, i] = np.where(ok, s[nb][:, i], -np.inf)
        mx0 = shfl_max(np.maximum(np.maximum(s[0][:, 0], s[0][:, 1]), np.maximum(s[1][:, 0], s[1][:, 1])))
        mx8 = np.full(32, NEG)
        if L > 1:
            mx8 = shfl_max(np.maximum(np.maximum(s[0][:, 2], s[0][:, 3]), np.maximum(s[1][:, 2], s[1][:, 3])))
        up0, up8 = mx0 > a.m0 + TAU, mx8 > a.m8 + TAU
        if (up0 | up8).any():
            al0 = np.where(up0, np.exp2(a.m0 - mx0), 1.0)
            al8 = np.where(up8, np.exp2(a.m8 - mx8), 1.0)
            a.m0 = np.where(up0, mx0, a.m0)
            a.m8 = np.where(up8, mx8, a.m8)
            a.o[:, :, 0:2] *= al0[None, :, None]
            a.o[:, :, 2:4] *= al8[None, :, None]
            a.l[:, 0:2] *= al0[:, None]
            a.l[:, 2:4] *= al8[:, None]

        def pp(x0, x1, m):
            with np.errstate(invalid="ignore"):
                if exp16:
                    d0, d1 = f16(x0 - m), f16(x1 - m)
                    return f16x2(np.exp2(d0), np.exp2(d1))
                return f16x2(np.exp2(x0 - m), np.exp2(x1 - m))

        pa = [pp(s[0][:, 0], s[0][:, 1], a.m0), None, pp(s[1][:, 0], s[1][:, 1], a.m0), None]
        if L > 1:
            pa[1] = pp(s[0][:, 2], s[0][:, 3], a.m8)
            pa[3] = pp(s[1][:, 2], s[1][:, 3], a.m8)
        else:
            pa[1] = pa[3] = np.zeros(32, dtype=np.uint32)
        for half in range(2):
            vr = ldsm_x4(vs, swz(kb * 16 + (mi & 1) * 8 + ri, h * 4 + half * 2 + (mi >> 1)), trans=True)
            for e in range(2):
                db = half * 2 + e
                lo, hi8 = vr[2 * e], vr[2 * e + 1]
                mma_f16(a.o[2 * db], pa, [e4m3x2_f16x2(byte_perm(lo, 0x0020)), e4m3x2_f16x2(byte_perm(hi8, 0x0020))])
                mma_f16(a.o[2 * db + 1], pa,
                        [e4m3x2_f16x2(byte_perm(lo, 0x0031)), e4m3x2_f16x2(byte_perm(hi8, 0x0031))])
        ones = np.full(32, 0x3C003C00, dtype=np.uint32)
        mma_f16(a.l, pa, [ones, ones])


def plan(seq_lens, window, n_ctas, min_chunk, w):
    lens = [min(s, window - 1) if window > 0 else s for s in seq_lens]
    total = sum(lens)
    chunk = max(min_chunk, -(-total // n_ctas))
    w0, w1 = w * chunk, min(w * chunk + chunk, total)
    segs, off = [], 0
    for b, (seq, ln) in enumerate(zip(seq_lens, lens)):
        if ln > 0:
            mine = off < w1 and off + ln > w0
            target = (off + ln - 1) // chunk - off // chunk + 1
        else:
            mine = min(off // chunk, n_ctas - 1) == w
            target = 1
        if mine:
            start = seq - ln
            a0, a1 = max(off, w0), min(off + ln, w1)
            segs.append(dict(b=b, lo=start + (a0 - off if ln else 0), hi=start + (a1 - off if ln else 0),
                             target=target, last=(ln == 0) or (off + ln <= w1)))
        off += ln
    return segs


def kernel(d, n_ctas, min_chunk=64, exp16=None, orden=None):
    L, B = d["L"], d["B"]
    exp16 = (L == 8) if exp16 is None else exp16
    window, rope = d["window"], d["rope"]
    kc, vc = d["kc"], d["vc"]   # [rows, 128] uint8, se escriben
    ks_, vs_ = d["k_scale"], d["v_scale"]
    scale_log2 = np.float32(QK * SM * LOG2E)
    c_ref = float(scale_log2) * 64 - HOLGURA
    scl = np.float32(scale_log2 * np.float32(ks_))
    acc = np.zeros((B, L, 16, 64))
    lsum = np.zeros((B, L, 16))
    cnt = np.zeros(B, dtype=int)
    out = np.zeros((B, L, 1024), dtype=np.float32)
    ctas = list(range(n_ctas)) if orden is None else orden
    for w in ctas:
        for sg in plan(d["seq_lens"], window, n_ctas, min_chunk, w):
            b, seq = sg["b"], d["seq_lens"][sg["b"]]
            n_tiles = -(-(sg["hi"] - sg["lo"]) // TILE)
            # q -> E4M3 [j][head][64]
            q_s = np.zeros(L * 16 * 64, dtype=np.uint8)
            for j in range(L):
                for head in range(16):
                    x = d["qkv"][b, j, head * 64:(head + 1) * 64].astype(np.float32)
                    x = x / np.sqrt(np.float32(np.mean(x.astype(np.float64) ** 2) + EPS))
                    if rope:
                        x = _rope(x, seq + j, d["cs"])
                    q_s[(j * 16 + head) * 64:(j * 16 + head + 1) * 64] = a_e4m3(x)
            warps = [Warp() for _ in range(8)]
            qas = []
            lane = np.arange(32)
            mi, ri = lane >> 3, lane & 7
            for warp in range(8):
                h = warp >> 2
                qa = []
                for ks in range(2):
                    if L == 1:
                        r = ldsm_x4(q_s, (h * 8 + ri) * 64 + (ks * 2 + (mi >> 1)) * 16)
                        z = np.zeros(32, dtype=np.uint32)
                        qa.append([r[0], z, r[2], z])
                    else:
                        rr = (mi & 1) * 8 + ri
                        tok = 2 * (warp & 3) + (rr >> 3)
                        qa.append(ldsm_x4(q_s, (tok * 16 + h * 8 + (rr & 7)) * 64 + (ks * 2 + (mi >> 1)) * 16))
                qas.append(qa)
            for it in range(n_tiles):
                st = np.zeros(2 * TILE * ROWB, dtype=np.uint8)
                for rc in range(TILE):
                    t = sg["lo"] + it * TILE + rc
                    if t < sg["hi"]:
                        r = _fila(d, b, t)
                        for c in range(8):
                            st[swz(rc, c): swz(rc, c) + 16] = kc[r, c * 16:(c + 1) * 16]
                            st[TILE * ROWB + swz(rc, c): TILE * ROWB + swz(rc, c) + 16] = vc[r, c * 16:(c + 1) * 16]
                for warp in range(8):
                    attend_tile(L, exp16, warps[warp], qas[warp], st[:TILE * ROWB], st[TILE * ROWB:],
                                sg["lo"] + it * TILE, sg["hi"], seq, window, scl, 4, warp)
            if sg["last"]:
                st = np.zeros(2 * TILE * ROWB, dtype=np.uint8)
                for i in range(L):
                    loc = d["cache_loc"][b * L + i]
                    for hh in range(2):
                        k = d["qkv"][b, i, (16 + hh) * 64:(17 + hh) * 64].astype(np.float32)
                        v = d["qkv"][b, i, (18 + hh) * 64:(19 + hh) * 64].astype(np.float32)
                        k = k / np.sqrt(np.float32(np.mean(k.astype(np.float64) ** 2) + EPS))
                        if rope:
                            k = _rope(k, seq + i, d["cs"])
                        k8, v8 = a_e4m3(k / ks_), a_e4m3(v / vs_)
                        for dd in range(64):
                            byte = hh * 64 + dd
                            st[swz(i, byte >> 4) + (byte & 15)] = k8[dd]
                            st[TILE * ROWB + swz(i, byte >> 4) + (byte & 15)] = v8[dd]
                        kc[loc, hh * 64:(hh + 1) * 64] = k8
                        vc[loc, hh * 64:(hh + 1) * 64] = v8
                for warp in range(8):
                    attend_tile(L, exp16, warps[warp], qas[warp], st[:TILE * ROWB], st[TILE * ROWB:], seq, seq + L,
                                seq, window, scl, 1, warp)
            lane = np.arange(32)
            g, tig = lane >> 2, lane & 3
            for warp in range(8):
                a = warps[warp]
                h = warp >> 2
                jg = 0 if L == 1 else 2 * (warp & 3)
                for rows, m, cols, lcol in ((0, a.m0, (0, 1), 0), (1, a.m8, (2, 3), 2)):
                    if rows == 1 and L == 1:
                        continue
                    f = np.exp2(m - c_ref)
                    for t in range(32):
                        if m[t] <= NEG:
                            continue
                        head = h * 8 + g[t]
                        j = jg + rows
                        for db in range(4):
                            base = db * 16 + tig[t] * 4
                            vals = [a.o[2 * db][t, cols[0]], a.o[2 * db + 1][t, cols[0]], a.o[2 * db][t, cols[1]],
                                    a.o[2 * db + 1][t, cols[1]]]
                            acc[b, j, head, base:base + 4] += np.array(vals) * f[t]
                        if tig[t] == 0:
                            lsum[b, j, head] += a.l[t, lcol] * f[t]
            cnt[b] += 1
            if cnt[b] == sg["target"]:
                for j in range(L):
                    for head in range(16):
                        o = acc[b, j, head] / lsum[b, j, head] * vs_
                        o = o / np.sqrt(np.mean(o ** 2) + EPS)
                        gate = d["qkv"][b, j, (20 + head) * 64:(21 + head) * 64].astype(np.float64)
                        out[b, j, head * 64:(head + 1) * 64] = bf16(o / (1 + np.exp(-gate)))
                acc[b] = 0
                lsum[b] = 0
                cnt[b] = 0
    assert not cnt.any() and not acc.any(), "algun pedido no cerro"
    return out


def _fila(d, b, t):
    page = d["page"]
    return int(d["table"][b, t // page]) * page + t % page


def _rope(x, pos, cs):
    c, s = cs[pos, :32], cs[pos, 32:]
    e, o = x[0::2].copy(), x[1::2].copy()
    y = np.empty_like(x)
    y[0::2] = e * c - o * s
    y[1::2] = e * s + o * c
    return y.astype(np.float32)


# ---- entradas y referencia ---------------------------------------------------------------------------------------------
def entradas(rng, B, L, ctx, layer, page, kv_scale=1.0, dist="normal", spread=False):
    seq_lens = [int(ctx * (rng.uniform(0.3, 1.5) if spread else 1)) for _ in range(B)]
    seq_lens = [max(s, 1) for s in seq_lens]
    pages_per = [(s + L + page - 1) // page for s in seq_lens]
    max_pages = max(pages_per)
    total = B * max_pages
    perm = rng.permutation(total)
    table = np.full((B, max_pages), -1, dtype=np.int64)
    o = 0
    for b in range(B):
        table[b, :pages_per[b]] = perm[o:o + pages_per[b]]
        o += pages_per[b]
    rows = total * page
    k = rng.standard_normal((rows, 2, 64))
    if dist == "soft":
        u = rng.standard_normal((2, 64))
        u /= np.sqrt((u ** 2).mean(-1, keepdims=True))
        k = np.sqrt(0.6) * u + np.sqrt(0.4) * k
    kc = a_e4m3(k / kv_scale).reshape(rows, 128)
    vc = a_e4m3(rng.standard_normal((rows, 2, 64)) / kv_scale).reshape(rows, 128)
    q = 2 * rng.standard_normal((B, L, 16 * 64))
    kn = 2 * rng.standard_normal((B, L, 2 * 64))
    vn = rng.standard_normal((B, L, 2 * 64))
    gate = rng.standard_normal((B, L, 16 * 64))
    qkv = bf16(np.concatenate([q, kn, vn, gate], -1))
    if dist == "edges" and layer == "sliding":   # llaves alineadas con la consulta en los bordes de la ventana
        for b in range(B):
            for t in (seq_lens[b] - WINDOW + 1, seq_lens[b] - WINDOW):
                if 0 <= t < seq_lens[b]:
                    for hh in range(2):
                        x = qkv[b, 0, hh * 8 * 64:(hh * 8 + 1) * 64].astype(np.float32)
                        kc[_fila(dict(page=page, table=table), b, t), hh * 64:(hh + 1) * 64] = a_e4m3(
                            x / np.sqrt(np.mean(x ** 2)) / kv_scale)
    cache_loc = np.array([_fila(dict(page=page, table=table), b, seq_lens[b] + i) for b in range(B) for i in range(L)])
    inv = 1.0 / (THETA ** (np.arange(0, 64, 2, dtype=np.float32) / 64)).astype(np.float32)
    pos = np.arange(max(seq_lens) + L + 1, dtype=np.float32)
    fr = (pos[:, None] * inv[None, :]).astype(np.float32)
    cs = np.concatenate([np.cos(fr), np.sin(fr)], -1).astype(np.float32)
    return dict(B=B, L=L, seq_lens=seq_lens, table=table, page=page, kc=kc, vc=vc, k_scale=np.float32(kv_scale),
                v_scale=np.float32(kv_scale), qkv=qkv.reshape(B, L, -1), cache_loc=cache_loc,
                window=WINDOW if layer == "sliding" else 0, rope=layer == "sliding", cs=cs)


def referencia(d, kc, vc):
    """float64; consultas redondeadas a E4M3 como FA3; las piezas nuevas las que quedaron en el cache."""
    B, L = d["B"], d["L"]
    out = np.zeros((B, L, 1024))
    for b in range(B):
        seq = d["seq_lens"][b]
        rows = [_fila(d, b, t) for t in range(seq + L)]
        K = E4M3[kc[rows]].reshape(-1, 2, 64) * d["k_scale"]
        V = E4M3[vc[rows]].reshape(-1, 2, 64) * d["v_scale"]
        for j in range(L):
            p = seq + j
            lo = max(0, p - d["window"] + 1) if d["window"] else 0
            for head in range(16):
                x = d["qkv"][b, j, head * 64:(head + 1) * 64].astype(np.float64)
                x = x / np.sqrt(np.mean(x ** 2) + EPS)
                if d["rope"]:
                    x = _rope(x.astype(np.float32), p, d["cs"]).astype(np.float64)
                x = E4M3[a_e4m3(x)] * QK
                hh = head // 8
                sc = K[lo:p + 1, hh] @ x * SM
                w = np.exp(sc - sc.max())
                w /= w.sum()
                o = w @ V[lo:p + 1, hh]
                o = o / np.sqrt(np.mean(o ** 2) + EPS)
                gate = d["qkv"][b, j, (20 + head) * 64:(21 + head) * 64].astype(np.float64)
                out[b, j, head * 64:(head + 1) * 64] = o / (1 + np.exp(-gate))
    return out


def main():
    rng = np.random.default_rng(7)
    casos = [  # (B, L, ctx, layer, page, kv_scale, dist, spread, n_ctas)
        (1, 1, 10, "sliding", 64, 1.0, "normal", False, 4),
        (3, 1, 300, "full", 1, 1.0, "soft", True, 5),
        (2, 8, 200, "full", 64, 1 / 64, "normal", True, 3),
        (2, 8, 2200, "sliding", 1, 1.0, "edges", False, 7),
        (3, 1, 2100, "sliding", 1, 1.0, "edges", True, 9),
        (1, 8, 130, "full", 1, 1.0, "soft", False, 2),
        (4, 1, 40, "full", 64, 1.0, "normal", True, 64),
    ]
    if "--rapido" in sys.argv:
        casos = casos[:2]
    peor = 0.0
    for B, L, ctx, layer, page, kvs, dist, spread, n_ctas in casos:
        d = entradas(rng, B, L, ctx, layer, page, kvs, dist, spread)
        kc0, vc0 = d["kc"].copy(), d["vc"].copy()
        orden = list(rng.permutation(n_ctas))
        out = kernel(d, n_ctas, orden=orden)
        ref = referencia(d, d["kc"], d["vc"])
        err = np.linalg.norm(out - ref) / np.linalg.norm(ref)
        fila = np.linalg.norm(out - ref, axis=-1) / np.linalg.norm(ref, axis=-1)
        # las filas nuevas del cache contra las de la referencia (k: normalizada, rotada, entre la escala)
        cambio = (d["kc"] != kc0).any(1).sum() + (d["vc"] != vc0).any(1).sum()
        print(f"B={B} L={L} ctx={ctx} {layer:7s} pag={page:2d} kv={kvs:.4f} {dist:6s} ctas={n_ctas:3d}: "
              f"L2 rel {err:.2e} (peor fila {fila.max():.2e}); filas del cache escritas {cambio}")
        peor = max(peor, err)
    print("peor", peor)
    sys.exit(0 if peor < 0.02 else 1)


if __name__ == "__main__":
    main()
