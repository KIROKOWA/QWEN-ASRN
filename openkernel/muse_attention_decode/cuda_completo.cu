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
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(16) __half q_s[16][64];  // las 16 cabezas de consulta, normalizadas y por qk_escala
    const int s = blockIdx.x, b = blockIdx.y;
    const int hilo = threadIdx.x, warp = hilo >> 5, carril = hilo & 31;
    const int g = carril >> 2, tig = carril & 3;   // grupo y posicion en el grupo (fragmentos de mma.sync)
    // v13: cada warp atiende UNA cabeza kv (h) y 16 de las 64 llaves de la vuelta (kg): la mitad de acumuladores y
    // de fragmentos de q que en la v12 (que llevaba las dos cabezas por warp), para que quepan 3 bloques por SM.
    const int h = warp >> 2, kg = warp & 3;

    const int seq = seq_lens[b];
    const int inicio = s * chunk;
    const int fin = min(inicio + chunk, seq);
    const int n_vueltas = fin > inicio ? (fin - inicio + LLAVES - 1) / LLAVES : 0;

    // Que copia este hilo: la fila hilo / 4 de la vuelta, cuarto (hilo & 3) de los 128 bytes, de k y de v.
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

    // Prologo: las primeras ETAPAS - 1 vueltas en camino mientras se normalizan las consultas.
    int fila_sig = 0;
#pragma unroll
    for (int v = 0; v < ETAPAS - 1; ++v) {
        if (v < n_vueltas) pedir(v, fila_de(v));
        cp_async_commit();
    }
    if (ETAPAS - 1 < n_vueltas) fila_sig = fila_de(ETAPAS - 1);

    if (hilo < 128) {   // RMSNorm de las 16 cabezas de consulta (8 hilos por cabeza, 8 valores cada uno), en f16.
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

    // Los fragmentos A de q se leen de q_s en cada vuelta (16 lecturas de 4 bytes) en vez de vivir en 8 registros
    // durante todo el bucle: con 3 bloques por SM, cada registro cuenta.
    const float escala = k_scale[0] * escala_log2;

    // Acumulado traspuesto, o^T = v^T p^T: las 16 filas del tensor core son 16 dimensiones de v (sin relleno), las 8
    // columnas las 8 consultas. 4 bloques de 16 dimensiones: dimension md * 16 + g (y + 8), consultas tig*2, tig*2+1.
    float o[4][4];
    float l = 0.f;
#pragma unroll
    for (int md = 0; md < 4; ++md)
#pragma unroll
        for (int i = 0; i < 4; ++i) o[md][i] = 0.f;

    const int llave0 = kg * 16;                        // las 16 llaves de este warp en cada vuelta
#pragma unroll 1
    for (int vuelta = 0; vuelta < n_vueltas; ++vuelta) {
        cp_async_wait<ETAPAS - 2>();
        __syncthreads();   // la vuelta llego para todos, y la ranura de la vuelta anterior ya nadie la lee
        if (vuelta + ETAPAS - 1 < n_vueltas) pedir(vuelta + ETAPAS - 1, fila_sig);
        cp_async_commit();
        if (vuelta + ETAPAS < n_vueltas) fila_sig = fila_de(vuelta + ETAPAS);

        const uint8_t* ks = smem + (vuelta % ETAPAS) * BLOQUE_ETAPA;
        const uint8_t* vs = ks + LLAVES * FILA;
        const int t0 = inicio + vuelta * LLAVES + llave0;
        const int gv = opaco(g), tv = opaco(tig);  // ver opaco: menos registros vivos a cambio de unas sumas
        // s = q . k: 2 bloques de 8 llaves x 4 pasos de 16 dimensiones; p = 2^(s - c), 0 fuera del pedazo.
        uint32_t pa[2];
#pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
            float sc[4] = {0.f, 0.f, 0.f, 0.f};
            const int fila = llave0 + nb * 8 + gv;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const int byte = h * 64 + kk * 16 + tv * 2;
                const __half* qf = &q_s[h * 8 + gv][kk * 16 + tv * 2];
                mma_f16(sc, *reinterpret_cast<const uint32_t*>(qf), *reinterpret_cast<const uint32_t*>(qf + 8),
                        e4m3x2_a_f16x2(*reinterpret_cast<const uint16_t*>(ks + lugar(fila, byte))),
                        e4m3x2_a_f16x2(*reinterpret_cast<const uint16_t*>(ks + lugar(fila, byte + 8))));
            }
            const int t = t0 + nb * 8 + tv * 2;
            const float p0 = t < fin ? exp2_rapido(fminf(sc[0] * escala - c_ref, 100.f)) : 0.f;
            const float p1 = t + 1 < fin ? exp2_rapido(fminf(sc[1] * escala - c_ref, 100.f)) : 0.f;
            l += p0 + p1;
            pa[nb] = f32x2_a_bf16x2(p0, p1);
        }
        // o^T += v^T p^T: A = v^T (16 dimensiones x las 16 llaves de este warp), B = p^T, que es justo el fragmento C
        // de q.k que este hilo ya tiene (llaves tig*2, tig*2+1 de cada bloque de 8, consulta g): sin mover datos.
        const int fv0 = llave0 + tv * 2, fv8 = llave0 + 8 + tv * 2;
#pragma unroll
        for (int md = 0; md < 4; ++md) {
            const int byte = h * 64 + md * 16 + gv;
            mma_bf16(o[md], v_par(vs, fv0, byte), v_par(vs, fv0, byte + 8), v_par(vs, fv8, byte),
                     v_par(vs, fv8, byte + 8), pa[0], pa[1]);
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    // Juntar los 4 warps de cada cabeza (cada uno vio 16 de cada 64 llaves) en la memoria compartida, ya libre.
    float* red_o = reinterpret_cast<float*>(smem);            // [4 grupos de llaves][2 cabezas][8 filas][64]
    float* red_l = red_o + 4 * 2 * 8 * 64;                     // [4][2][8]
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    if (tig == 0) red_l[(kg * 2 + h) * 8 + g] = l;
#pragma unroll
    for (int md = 0; md < 4; ++md) {   // o^T[dimension][consulta] -> red_o[grupo][cabeza][consulta][dimension]
        float* d = red_o + ((kg * 2 + h) * 8 + tig * 2) * 64 + md * 16 + g;
        d[0] = o[md][0];        // (dim md*16+g,   consulta tig*2)
        d[64] = o[md][1];       // (dim md*16+g,   consulta tig*2+1)
        d[8] = o[md][2];        // (dim md*16+g+8, consulta tig*2)
        d[64 + 8] = o[md][3];   // (dim md*16+g+8, consulta tig*2+1)
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

}  // namespace muse
