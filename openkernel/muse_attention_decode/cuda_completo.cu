// Atencion de un paso de decodificacion sobre el contexto completo (Muse Spark Open, TP8: 16 cabezas de consulta,
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
__device__ __forceinline__ int fila_cache(const int* tabla, int64_t paso_tabla, int b, int t, int pagina) {
    return tabla[b * paso_tabla + t / pagina] * pagina + t % pagina;
}

extern "C" __global__ void __launch_bounds__(HILOS, BLOQUES_POR_SM)
parcial_completo(const __nv_bfloat16* __restrict__ qkv, int64_t paso_qkv, const int* __restrict__ seq_lens,
                 const int* __restrict__ tabla, int64_t paso_tabla, int pagina, const uint8_t* __restrict__ kc,
                 const uint8_t* __restrict__ vc, const float* __restrict__ k_scale, float qk_escala,
                 float escala_log2, float c_ref, float* __restrict__ mp, float* __restrict__ lp,
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
        return t < fin ? fila_cache(tabla, paso_tabla, b, t, pagina) : 0;  // fuera del pedazo: la fila 0 (se enmascara)
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
            mp[donde] = c_ref;
        }
    }
}

// --- verify (q_len = 8): 64 filas de consulta por cabeza kv (8 piezas x 8 cabezas) --------------------------------------
// Cada warp toma una cabeza kv (h) y 16 filas REALES (grupo rg: las piezas 2rg y 2rg+1), y todas las 64 llaves de cada
// vuelta: no hay suma entre warps, y p pasa del fragmento C de q.k al fragmento A de p.v en registros (como FA2). Los
// fragmentos B de v salen de vb con ldmatrix.trans. Fila r del pedazo = pieza r / 8, cabeza de consulta h * 8 + r % 8.
extern "C" __global__ void __launch_bounds__(HILOS, BLOQUES_POR_SM)
parcial_completo_verify(const __nv_bfloat16* __restrict__ qkv, int64_t paso_qkv, const int* __restrict__ seq_lens,
                        const int* __restrict__ tabla, int64_t paso_tabla, int pagina, const uint8_t* __restrict__ kc,
                        const uint8_t* __restrict__ vc, const float* __restrict__ k_scale, float qk_escala,
                        float escala_log2, float c_ref, float* __restrict__ mp, float* __restrict__ lp,
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
        return t < fin ? fila_cache(tabla, paso_tabla, b, t, pagina) : 0;
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
        mp[base + r0] = c_ref;
        mp[base + r0 + 8] = c_ref;
    }
}

}  // namespace muse
