"""Simulador en numpy del kernel CUDA parcial_completo (cuda_completo.cu), carril por carril, para revisar sin GPU los
indices: los fragmentos de mma.sync m16n8k16 (formato de la documentacion de PTX), el swizzle de la memoria
compartida, el anillo de vueltas y la reduccion entre warps. Compara el pedazo que produce con un calculo directo.

    python simular_cuda.py

Los fragmentos de mma.sync.m16n8k16.row.col (g = carril // 4, tig = carril % 4):
  A (16 x 16): a0 = (g, 2tig..2tig+1), a1 = (g+8, 2tig..), a2 = (g, 2tig+8..), a3 = (g+8, 2tig+8..)
  B (16 x 8):  b0 = (k = 2tig..2tig+1, n = g), b1 = (k = 2tig+8..2tig+9, n = g)
  C (16 x 8):  c0, c1 = (g, 2tig..2tig+1), c2, c3 = (g+8, 2tig..2tig+1)"""

import numpy as np
import torch

LLAVES, FILA, ETAPAS = 64, 128, 4


def mma(a, b, c):
    """a: [32 carriles][4][2] (pares de A), b: [32][2][2] (pares de B), c: [32][4]; devuelve D = A B + C por carril."""
    A = np.zeros((16, 16), np.float64)
    B = np.zeros((16, 8), np.float64)
    for carril in range(32):
        g, tig = carril // 4, carril % 4
        for i, (fila, k0) in enumerate([(g, 2 * tig), (g + 8, 2 * tig), (g, 2 * tig + 8), (g + 8, 2 * tig + 8)]):
            A[fila, k0:k0 + 2] = a[carril][i]
        for i, k0 in enumerate([2 * tig, 2 * tig + 8]):
            B[k0:k0 + 2, g] = b[carril][i]
    D = A @ B
    out = np.array(c, np.float64).copy()
    for carril in range(32):
        g, tig = carril // 4, carril % 4
        out[carril] += [D[g, 2 * tig], D[g, 2 * tig + 1], D[g + 8, 2 * tig], D[g + 8, 2 * tig + 1]]
    return out


def lugar(fila, byte):
    return fila * FILA + ((((byte >> 4) ^ (fila & 7)) << 4) | (byte & 15))


def e4m3(bytes_):
    return torch.from_numpy(np.asarray(bytes_, np.uint8)).view(torch.float8_e4m3fn).float().numpy().astype(np.float64)


def simular(qkv_b, kc, vc, tabla_b, pagina, seq, s, chunk, k_scale, qk_escala, escala_log2, c_ref):
    """El pedazo s de una peticion: devuelve (l [2, 8], acc [2, 8, 64]) como los escribiria el kernel."""
    inicio, fin = s * chunk, min(s * chunk + chunk, seq)
    n_vueltas = (fin - inicio + LLAVES - 1) // LLAVES if fin > inicio else 0
    smem = np.zeros(ETAPAS * 2 * LLAVES * FILA, np.uint8)
    # Q: RMSNorm por qk_escala, redondeada a bf16, guardada en f16 (como el kernel).
    x = qkv_b[:16 * 64].reshape(16, 64).astype(np.float64)
    q = x / np.sqrt((x ** 2).mean(1, keepdims=True) + 1e-5) * qk_escala
    q = torch.tensor(q).to(torch.bfloat16).to(torch.float16).double().numpy()
    escala = k_scale * escala_log2

    def pedir(vuelta):  # lo que hacen los 128 hilos con cp.async
        base = (vuelta % ETAPAS) * 2 * LLAVES * FILA
        for hilo in range(128):
            fila_c, mitad = hilo >> 1, hilo & 1
            t = inicio + vuelta * LLAVES + fila_c
            fila = int(tabla_b[t // pagina]) * pagina + t % pagina if t < fin else 0
            for c in range(4):
                byte = mitad * 64 + c * 16
                smem[base + lugar(fila_c, byte):base + lugar(fila_c, byte) + 16] = kc[fila, byte:byte + 16]
                d = base + LLAVES * FILA + lugar(fila_c, byte)
                smem[d:d + 16] = vc[fila, byte:byte + 16]

    for v in range(min(ETAPAS - 1, n_vueltas)):
        pedir(v)
    o = np.zeros((4, 32, 2, 8, 4))
    l = np.zeros((4, 32, 2))
    for vuelta in range(n_vueltas):
        if vuelta + ETAPAS - 1 < n_vueltas:
            pedir(vuelta + ETAPAS - 1)
        ks = (vuelta % ETAPAS) * 2 * LLAVES * FILA
        vs = ks + LLAVES * FILA
        for warp in range(4):
            llave0 = warp * 16
            t0 = inicio + vuelta * LLAVES + llave0
            for h in range(2):
                sc = []
                for nb in range(2):
                    c = np.zeros((32, 4))
                    for kk in range(4):
                        a, b = [], []
                        for carril in range(32):
                            g, tig = carril // 4, carril % 4
                            fila = llave0 + nb * 8 + g
                            byte = h * 64 + kk * 16 + tig * 2
                            a.append([q[h * 8 + g, kk * 16 + tig * 2:kk * 16 + tig * 2 + 2], [0, 0],
                                      q[h * 8 + g, kk * 16 + 8 + tig * 2:kk * 16 + 8 + tig * 2 + 2], [0, 0]])
                            b.append([e4m3(smem[ks + lugar(fila, byte):ks + lugar(fila, byte) + 2]),
                                      e4m3(smem[ks + lugar(fila, byte + 8):ks + lugar(fila, byte + 8) + 2])])
                        c = mma(a, b, c)
                    sc.append(c)
                p = np.zeros((32, 2, 2))
                for carril in range(32):
                    g, tig = carril // 4, carril % 4
                    for nb in range(2):
                        for i in range(2):
                            t = t0 + nb * 8 + tig * 2 + i
                            e = min(sc[nb][carril][i] * escala - c_ref, 100.0)
                            p[carril, nb, i] = 2.0 ** e if t < fin else 0.0
                            l[warp, carril, h] += p[carril, nb, i]
                pb = torch.tensor(p).to(torch.bfloat16).double().numpy()
                for nd in range(8):
                    a, b = [], []
                    for carril in range(32):
                        g, tig = carril // 4, carril % 4
                        fv0, fv8 = llave0 + tig * 2, llave0 + 8 + tig * 2
                        byte = h * 64 + nd * 8 + g
                        a.append([pb[carril, 0], [0, 0], pb[carril, 1], [0, 0]])
                        b.append([e4m3([smem[vs + lugar(fv0, byte)], smem[vs + lugar(fv0 + 1, byte)]]),
                                  e4m3([smem[vs + lugar(fv8, byte)], smem[vs + lugar(fv8 + 1, byte)]])])
                    o[warp, :, h, nd] = mma(a, b, o[warp, :, h, nd])
    # Reduccion entre warps (como el kernel).
    L = np.zeros((2, 8))
    ACC = np.zeros((2, 8, 64))
    for warp in range(4):
        for carril in range(32):
            g, tig = carril // 4, carril % 4
            for h in range(2):
                L[h, g] += l[warp, carril, h]
                for nd in range(8):
                    ACC[h, g, nd * 8 + tig * 2:nd * 8 + tig * 2 + 2] += o[warp, carril, h, nd, :2]
    return L, ACC


def directo(qkv_b, kc, vc, tabla_b, pagina, seq, s, chunk, k_scale, qk_escala, escala_log2, c_ref):
    """El mismo pedazo calculado directo, en float64."""
    inicio, fin = s * chunk, min(s * chunk + chunk, seq)
    x = qkv_b[:16 * 64].reshape(16, 64).astype(np.float64)
    q = x / np.sqrt((x ** 2).mean(1, keepdims=True) + 1e-5) * qk_escala
    q = torch.tensor(q).to(torch.bfloat16).double().numpy()
    t = np.arange(inicio, fin)
    filas = tabla_b[t // pagina].astype(np.int64) * pagina + t % pagina
    K = e4m3(kc[filas]).reshape(-1, 2, 64)
    V = e4m3(vc[filas]).reshape(-1, 2, 64)
    L = np.zeros((2, 8))
    ACC = np.zeros((2, 8, 64))
    for h in range(2):
        sc = q[h * 8:(h + 1) * 8] @ K[:, h].T * k_scale * escala_log2 - c_ref
        p = 2.0 ** np.minimum(sc, 100.0)
        L[h] = p.sum(1)
        ACC[h] = p @ V[:, h]
    return L, ACC


def main():
    rng = np.random.default_rng(0)
    pagina, seq, chunk = 1, 300, 160
    n_filas = 512
    kc = torch.randn(n_filas, 128).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    vc = torch.randn(n_filas, 128).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    tabla_b = rng.permutation(n_filas)[:seq + 1].astype(np.int32)
    qkv_b = (2 * rng.standard_normal(2304)).astype(np.float32)
    args = dict(qkv_b=qkv_b, kc=kc, vc=vc, tabla_b=tabla_b, pagina=pagina, seq=seq, chunk=chunk, k_scale=1.0,
                qk_escala=5.4730064863875, escala_log2=0.125 * 1.4426950408889634)
    args["c_ref"] = args["qk_escala"] * 64 * args["escala_log2"] - 40.0
    peor = 0.0
    for s in (0, 1, 2):  # el 1 termina a mitad de una vuelta; el 2 no tiene llaves
        L1, A1 = simular(s=s, **args)
        L2, A2 = directo(s=s, **args)
        err_l = np.abs(L1 - L2).max() / max(np.abs(L2).max(), 1e-30)
        err_a = np.abs(A1 - A2).max() / max(np.abs(A2).max(), 1e-30)
        peor = max(peor, err_l, err_a)
        print(f"pedazo {s}: llaves {max(0, min(s * chunk + chunk, seq) - s * chunk)}, error relativo l {err_l:.2e}, "
              f"acc {err_a:.2e}")
    print("indices bien" if peor < 2e-2 else "INDICES MAL")


if __name__ == "__main__":
    main()
