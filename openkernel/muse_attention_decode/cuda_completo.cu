// Atencion de un paso de decodificacion sobre el contexto completo (Muse Spark Open, TP8: 16 cabezas de consulta,
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
