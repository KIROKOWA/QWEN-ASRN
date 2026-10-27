# okc: problem=muse_attention_decode
"""RelativisticRGAttention (RRGA) - El opuesto exacto al kernel cuantico de calibre fijo.
Fundamentos:
1. Relatividad Especial y Geometria Tropical (Maslov Dequantization): Cada segmento temporal
   opera en su propio marco inercial local m_s = max(s), sin energia de referencia fija global.
2. Grupo de Renormalizacion de Wilson (Flujo RG Libre de Atomicas): Cero atomicAdd en DRAM.
   Los marcos locales se acoplan mediante transformaciones de dilatacion Lorentz-tropicales
   exactas: gamma = 2^(m_local - m_global) en (0, 1].
3. Reloj Simplectico SO(2) en Registros (Cero Tablas en HBM): Elimina la tabla _ROPE de 536 MB
   calculando el propagador unitario directamente en los registros con reduccion de fase exacta.
4. Cobertura Total: Ejecuta Decode (L=1) y Verify (L=8), tanto en contexto completo (window=0)
   como en ventana deslizante (window=2048 con RoPE intercalado) en CUDA nativo.
"""

from typing import TypedDict
import torch


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 (decode) u 8 (verify causal)
    num_q_heads: int      # Hq: 16
    num_kv_heads: int     # Hkv: 2
    head_dim: int         # 64
    page_size: int        # 1 o 64
    window: int           # 2048 o 0
    rope_theta: float     # 500000.0 o 0.0
    qk_scale_factor: float
    sm_scale: float       # 1/8
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


LOG2E = 1.4426950408889634
_ESPACIO_RG = {}
_MODULO_RG = {}


def _buffers_rg(device: torch.device, n_bh: int, max_splits: int, max_filas: int):
    """Buffers deterministas disjuntos para el flujo del Grupo de Renormalizacion (sin memset ni atomicas)."""
    clave = (str(device), n_bh, max_splits, max_filas)
    buf = _ESPACIO_RG.get(clave)
    if buf is None:
        m_buf = torch.empty((n_bh, max_splits, max_filas), dtype=torch.float32, device=device)
        z_buf = torch.empty((n_bh, max_splits, max_filas), dtype=torch.float32, device=device)
        o_buf = torch.empty((n_bh, max_splits, max_filas, 64), dtype=torch.float32, device=device)
        buf = _ESPACIO_RG[clave] = (m_buf, z_buf, o_buf)
    return buf


_CUDA_RELATIVISTIC_RG = r"""
#ifdef __CUDACC_RTC__
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

namespace relativistic_rg {

constexpr int HILOS_SCAN = 128;          // 4 warps por bloque de escaneo local
constexpr int LLAVES_PASO = 64;          // 64 llaves por ciclo
constexpr int BYTES_FILA_CACHE = 128;    // 2 cabezas KV x 64 bytes
constexpr int ETAPAS_TUBO = 3;           // Pipeline asincrono cp.async
constexpr int BYTES_ETAPA = 2 * LLAVES_PASO * BYTES_FILA_CACHE;
constexpr float VACIO_TROPICAL = -1e30f; // Identidad aditiva del semianillo tropical (max, +)

__device__ __forceinline__ uint32_t a_smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ int swizzle_fase(int fila, int byte) {
    return fila * BYTES_FILA_CACHE + ((((byte >> 4) ^ (fila & 7)) << 4) | (byte & 15));
}

__device__ __forceinline__ float exp2_exacto(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint8_t a_e4m3_par(float x) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float e4m3_a_f32(uint8_t b) {
    __half_raw h = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

// Reloj Simplectico SO(2) en Registros: Calcula (cos, sin) sin tablas en HBM.
// Usa particion temporal de Sterbenz (t = t_hi + t_lo) para evitar perdida de fase a grandes posiciones.
__device__ __forceinline__ void reloj_simplectico_so2(int pos, int par_idx, float rope_theta, float& c_out, float& s_out) {
    if (rope_theta <= 0.f) {
        c_out = 1.f;
        s_out = 0.f;
        return;
    }
    // inv_freq = theta^(-2 * par_idx / 64) = 2^(-2 * par_idx / 64 * log2(theta))
    const float exponente = -static_cast<float>(2 * par_idx) * (1.f / 64.f) * log2f(rope_theta);
    const float inv_freq = exp2_exacto(exponente);
    // Particion de Sterbenz del reloj de posicion (alta y baja precision)
    const int pos_hi = pos & ~1023;
    const int pos_lo = pos & 1023;
    const float fase = __fmaf_rn(static_cast<float>(pos_hi), inv_freq, static_cast<float>(pos_lo) * inv_freq);
    __sincosf(fase, &s_out, &c_out);
}

// Normalizacion RMS + Rotacion Unitaria Intercalada SO(2) para una fila de 64 dimensiones
// 2 hilos cooperan por cada fila de 64 elementos (cada hilo procesa 16 pares = 32 floats)
__device__ __forceinline__ void rms_y_reloj_intercalado(
    const __nv_bfloat16* fila_ptr, int pos, float escala, float rope_theta,
    int sub_mitad, float* par_out, float* impar_out) {

    float ss = 0.f;
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        const int idx_par = sub_mitad * 16 + m;
        const float p = __bfloat162float(fila_ptr[2 * idx_par]);
        const float im = __bfloat162float(fila_ptr[2 * idx_par + 1]);
        par_out[m] = p;
        impar_out[m] = im;
        ss += p * p + im * im;
    }
    // Reduccion entre los 2 hilos que comparten la fila (xor 1)
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    const float inv = rsqrtf(ss * (1.f / 64.f) + 1e-5f) * escala;

#pragma unroll
    for (int m = 0; m < 16; ++m) {
        const int idx_par = sub_mitad * 16 + m;
        float p = par_out[m] * inv;
        float im = impar_out[m] * inv;
        if (rope_theta > 0.f) {
            float c, s;
            reloj_simplectico_so2(pos, idx_par, rope_theta, c, s);
            const float p_rot = p * c - im * s;
            const float im_rot = p * s + im * c;
            p = p_rot;
            im = im_rot;
        }
        par_out[m] = p;
        impar_out[m] = im;
    }
}

__device__ __forceinline__ int resolver_fila_pagina(const int* tabla, int64_t paso_tabla, int b, int t, int pagina) {
    if (pagina == 1) return tabla[b * paso_tabla + t];
    return tabla[b * paso_tabla + t / pagina] * pagina + (t % pagina);
}

// ====================================================================================================================
// ETAPA 1: Escaneo Relativista en Marcos Inerciales Locales (Álgebra Tropical Max-Plus)
// Malla: (n_splits, B * Hkv). Cero atomicas. Cada bloque evalua un intervalo [inicio, fin) de llaves
// y emite su triplete del Grupo de Renormalizacion (m_local, Z_local, O_local) en su propio marco de referencia.
// Soporta tanto Decode (L=1, 8 filas por cabeza KV) como Verify (L=8, 64 filas por cabeza KV).
// ====================================================================================================================
extern "C" __global__ void __launch_bounds__(HILOS_SCAN, 2)
rg_etapa1_marcos_locales(
    const __nv_bfloat16* __restrict__ qkv, int64_t paso_qkv,
    const int* __restrict__ seq_lens,
    const int* __restrict__ tabla, int64_t paso_tabla, int pagina,
    const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale,
    float* __restrict__ m_out, float* __restrict__ z_out, float* __restrict__ o_out,
    int chunk, int n_splits, int L, int hq, int hkv, int ventana,
    float rope_theta, float qk_escala, float escala_log2) {

    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(16) float q_sh[64][64]; // Hasta 64 filas (L=8 x G=8) x 64 dims

    const int s = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / hkv;
    const int h = bh % hkv;
    const int g_heads = hq / hkv; // 8
    const int n_filas = L * g_heads; // 8 en decode, 64 en verify
    const int tid = threadIdx.x;

    const int seq = seq_lens[b];
    const int lo_global = (ventana > 0) ? max(seq - ventana + 1, 0) : 0;
    const int inicio = lo_global + s * chunk;
    const int fin = min(inicio + chunk, seq);

    // Si este split cae fuera del cono de luz pasado, emite el elemento neutro del monoide tropical y termina
    if (inicio >= fin) {
        for (int r = tid; r < n_filas; r += HILOS_SCAN) {
            const int64_t base_idx = (static_cast<int64_t>(bh) * n_splits + s) * n_filas + r;
            m_out[base_idx] = VACIO_TROPICAL;
            z_out[base_idx] = 0.f;
        }
        return;
    }

    // 1. Preparar las consultas Q en su marco propio con el Reloj Simplectico SO(2)
    // 2 hilos por fila -> 128 hilos procesan hasta 64 filas en 1 sola pasada
    const int fila_q_idx = tid >> 1;
    const int mitad_q = tid & 1;
    if (fila_q_idx < n_filas) {
        const int j = fila_q_idx / g_heads;
        const int g = fila_q_idx % g_heads;
        const int cabeza_q = h * g_heads + g;
        const int pos = seq + j;
        const __nv_bfloat16* ptr_q = qkv + static_cast<int64_t>(b * L + j) * paso_qkv + cabeza_q * 64;
        float par[16], impar[16];
        rms_y_reloj_intercalado(ptr_q, pos, qk_escala, rope_theta, mitad_q, par, impar);
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const int idx_par = mitad_q * 16 + m;
            q_sh[fila_q_idx][2 * idx_par] = par[m];
            q_sh[fila_q_idx][2 * idx_par + 1] = impar[m];
        }
    }
    __syncthreads();

    const float escala = k_scale[0] * escala_log2;
    const int n_vueltas = (fin - inicio + LLAVES_PASO - 1) / LLAVES_PASO;
    const int fila_carga = tid >> 1; // 0..63
    const int lado_carga = tid & 1;  // 0..1 (64 bytes cada lado)

    auto pedir_etapa = [&](int vuelta, int fila_fisica) {
        const int64_t f64 = fila_fisica;
        uint8_t* base = smem + (vuelta % ETAPAS_TUBO) * BYTES_ETAPA;
        const uint8_t* k_src = kc + f64 * BYTES_FILA_CACHE + lado_carga * 64;
        const uint8_t* v_src = vc + f64 * BYTES_FILA_CACHE + lado_carga * 64;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int byte = lado_carga * 64 + c * 16;
            cp_async16(a_smem(base + swizzle_fase(fila_carga, byte)), k_src + c * 16);
            cp_async16(a_smem(base + LLAVES_PASO * BYTES_FILA_CACHE + swizzle_fase(fila_carga, byte)), v_src + c * 16);
        }
    };

    auto fila_de_vuelta = [&](int vuelta) -> int {
        const int t = inicio + vuelta * LLAVES_PASO + fila_carga;
        return (t < fin) ? resolver_fila_pagina(tabla, paso_tabla, b, t, pagina) : 0;
    };

    int fila_sig = 0;
#pragma unroll
    for (int v = 0; v < ETAPAS_TUBO - 1; ++v) {
        if (v < n_vueltas) pedir_etapa(v, fila_de_vuelta(v));
        cp_async_commit();
    }
    if (ETAPAS_TUBO - 1 < n_vueltas) fila_sig = fila_de_vuelta(ETAPAS_TUBO - 1);

    // Asignacion de trabajo dentro del bloque:
    // 128 hilos divididos en 8 grupos de 16 hilos.
    // Cada grupo de 16 hilos mantiene el marco inercial de un conjunto de filas r (1 fila por grupo en L=1; 8 en L=8).
    // Dentro del grupo de 16 hilos, cada hilo owns 4 dimensiones de salida (16 * 4 = 64 dims).
    const int grupo_fila = tid >> 4;   // 0..7
    const int carril_dim = tid & 15;   // 0..15 (cada uno maneja 4 dimensiones: carril_dim*4 .. carril_dim*4+3)
    const int pasos_r = n_filas / 8;   // 1 para Decode (L=1), 8 para Verify (L=8)

    float m_loc[8];
    float z_loc[8];
    float o_loc[8][4];
#pragma unroll
    for (int ir = 0; ir < 8; ++ir) {
        m_loc[ir] = VACIO_TROPICAL;
        z_loc[ir] = 0.f;
#pragma unroll
        for (int d = 0; d < 4; ++d) o_loc[ir][d] = 0.f;
    }

    for (int vuelta = 0; vuelta < n_vueltas; ++vuelta) {
        cp_async_wait<ETAPAS_TUBO - 2>();
        __syncthreads();
        if (vuelta + ETAPAS_TUBO - 1 < n_vueltas) pedir_etapa(vuelta + ETAPAS_TUBO - 1, fila_sig);
        cp_async_commit();
        if (vuelta + ETAPAS_TUBO < n_vueltas) fila_sig = fila_de_vuelta(vuelta + ETAPAS_TUBO);

        const uint8_t* ks = smem + (vuelta % ETAPAS_TUBO) * BYTES_ETAPA;
        const uint8_t* vs = ks + LLAVES_PASO * BYTES_FILA_CACHE;
        const int t_base = inicio + vuelta * LLAVES_PASO;
        const int n_validas = min(LLAVES_PASO, fin - t_base);

        for (int ir = 0; ir < pasos_r; ++ir) {
            const int r = ir * 8 + grupo_fila;
            const int j = r / g_heads;
            const int pos_q = seq + j;
            const int lim_ventana = (ventana > 0) ? (pos_q - ventana) : -1;

            // Procesamos las llaves del paso en sub-lotes de 16 (1 llave por cada uno de los 16 carriles)
            for (int k0 = 0; k0 < n_validas; k0 += 16) {
                const int k_idx = k0 + carril_dim;
                const int t_pos = t_base + k_idx;
                float s_k = VACIO_TROPICAL;
                if (k_idx < n_validas && t_pos > lim_ventana) {
                    float dot = 0.f;
#pragma unroll
                    for (int d = 0; d < 64; ++d) {
                        const uint8_t kb = ks[swizzle_fase(k_idx, h * 64 + d)];
                        dot += q_sh[r][d] * e4m3_a_f32(kb);
                    }
                    s_k = dot * escala;
                }

                // Maximo tropical de las 16 llaves mediante reduccion en registro entre los 16 carriles
                float m_lote = s_k;
#pragma unroll
                for (int off = 1; off < 16; off <<= 1) {
                    m_lote = fmaxf(m_lote, __shfl_xor_sync(0xffffffffu, m_lote, off));
                }

                if (m_lote > VACIO_TROPICAL * 0.5f) {
                    // Transformacion de Lorentz / Flujo RG al nuevo marco inercial m_nuevo
                    const float m_nuevo = fmaxf(m_loc[ir], m_lote);
                    const float gamma_prev = exp2_exacto(m_loc[ir] - m_nuevo);
                    const float p_mio = (s_k > VACIO_TROPICAL * 0.5f) ? exp2_exacto(s_k - m_nuevo) : 0.f;

                    float z_lote = p_mio;
#pragma unroll
                    for (int off = 1; off < 16; off <<= 1) {
                        z_lote += __shfl_xor_sync(0xffffffffu, z_lote, off);
                    }
                    z_loc[ir] = z_loc[ir] * gamma_prev + z_lote;
                    m_loc[ir] = m_nuevo;

                    // Acumulacion de valores V en las 4 dimensiones que posee este carril
                    const int d_base = carril_dim * 4;
                    float o0 = o_loc[ir][0] * gamma_prev;
                    float o1 = o_loc[ir][1] * gamma_prev;
                    float o2 = o_loc[ir][2] * gamma_prev;
                    float o3 = o_loc[ir][3] * gamma_prev;

                    const int tope_u = min(16, n_validas - k0);
                    const int base_lane = tid & 16; // 0 o 16 dentro del warp
                    for (int u = 0; u < tope_u; ++u) {
                        const float p_u = __shfl_sync(0xffffffffu, p_mio, base_lane + u);
                        if (p_u > 0.f) {
                            const int ku = k0 + u;
                            o0 += p_u * e4m3_a_f32(vs[swizzle_fase(ku, h * 64 + d_base + 0)]);
                            o1 += p_u * e4m3_a_f32(vs[swizzle_fase(ku, h * 64 + d_base + 1)]);
                            o2 += p_u * e4m3_a_f32(vs[swizzle_fase(ku, h * 64 + d_base + 2)]);
                            o3 += p_u * e4m3_a_f32(vs[swizzle_fase(ku, h * 64 + d_base + 3)]);
                        }
                    }
                    o_loc[ir][0] = o0;
                    o_loc[ir][1] = o1;
                    o_loc[ir][2] = o2;
                    o_loc[ir][3] = o3;
                }
            }
        }
    }
    cp_async_wait<0>();

    // Escritura determinista sin colisiones del estado renormalizado de este split
    for (int ir = 0; ir < pasos_r; ++ir) {
        const int r = ir * 8 + grupo_fila;
        const int64_t idx_base = (static_cast<int64_t>(bh) * n_splits + s) * n_filas + r;
        const int d_base = carril_dim * 4;
        o_out[idx_base * 64 + d_base + 0] = o_loc[ir][0];
        o_out[idx_base * 64 + d_base + 1] = o_loc[ir][1];
        o_out[idx_base * 64 + d_base + 2] = o_loc[ir][2];
        o_out[idx_base * 64 + d_base + 3] = o_loc[ir][3];
        if (carril_dim == 0) {
            m_out[idx_base] = m_loc[ir];
            z_out[idx_base] = z_loc[ir];
        }
    }
}

// ====================================================================================================================
// ETAPA 2: Punto Fijo del Grupo de Renormalizacion + Cono de Luz Causal Nuevo + Colapso RMSNorm/Gate
// Malla: (B * Hkv, L * G). Un bloque (64 hilos = 2 warps) por fila de consulta r = j * G + g.
// El bloque r == 0 de cada (b, h) cuantiza y escribe las L llaves y valores nuevos en k_cache / v_cache.
// ====================================================================================================================
extern "C" __global__ void __launch_bounds__(64, 4)
rg_etapa2_colapso_causal(
    const __nv_bfloat16* __restrict__ qkv, int64_t paso_qkv,
    const int* __restrict__ seq_lens,
    const int64_t* __restrict__ cache_loc,
    uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    const float* __restrict__ m_in, const float* __restrict__ z_in, const float* __restrict__ o_in,
    __nv_bfloat16* __restrict__ out,
    int n_splits, int L, int hq, int hkv,
    float rope_theta, float qk_escala, float escala_log2) {

    __shared__ float q_fila[64];
    __shared__ float k_nuevas[8][64];
    __shared__ float v_nuevas[8][64];

    const int bh = blockIdx.x;
    const int r = blockIdx.y; // 0 .. L*G - 1
    const int b = bh / hkv;
    const int h = bh % hkv;
    const int g_heads = hq / hkv;
    const int n_filas = L * g_heads;
    const int j = r / g_heads;
    const int g = r % g_heads;
    const int cabeza_q = h * g_heads + g;
    const int tid = threadIdx.x; // 0..63 (1 hilo por cada dimension d = 0..63)

    const int seq = seq_lens[b];
    const float ks = k_scale[0];
    const float vs = v_scale[0];
    const float escala = ks * escala_log2;

    // 1. Reduccion del Flujo del Grupo de Renormalizacion sobre todos los n_splits
    float m_acc = VACIO_TROPICAL;
    float z_acc = 0.f;
    float o_dim = 0.f;

    for (int s = 0; s < n_splits; ++s) {
        const int64_t idx = (static_cast<int64_t>(bh) * n_splits + s) * n_filas + r;
        const float ms = m_in[idx];
        if (ms > VACIO_TROPICAL * 0.5f) {
            const float zs = z_in[idx];
            const float os = o_in[idx * 64 + tid];
            const float m_nuevo = fmaxf(m_acc, ms);
            const float gamma_a = exp2_exacto(m_acc - m_nuevo);
            const float gamma_b = exp2_exacto(ms - m_nuevo);
            z_acc = z_acc * gamma_a + zs * gamma_b;
            o_dim = o_dim * gamma_a + os * gamma_b;
            m_acc = m_nuevo;
        }
    }

    // 2. Reconstruir Q de esta fila (hilos 0 y 1 cooperan) y las L llaves/valores nuevos del bloque causal
    if (tid < 2) {
        const __nv_bfloat16* ptr_q = qkv + static_cast<int64_t>(b * L + j) * paso_qkv + cabeza_q * 64;
        float par[16], impar[16];
        rms_y_reloj_intercalado(ptr_q, seq + j, qk_escala, rope_theta, tid, par, impar);
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const int idx_par = tid * 16 + m;
            // Igual que _final_v2: q pasa por bf16 antes del producto con k8
            q_fila[2 * idx_par] = __bfloat162float(__float2bfloat16_rn(par[m]));
            q_fila[2 * idx_par + 1] = __bfloat162float(__float2bfloat16_rn(impar[m]));
        }
    }

    // Los primeros 2*L hilos (hasta 16 hilos para L=8) normalizan y rotan las L llaves nuevas
    if (tid < 2 * L) {
        const int i_tok = tid >> 1;
        const int mitad = tid & 1;
        const int64_t tok_global = static_cast<int64_t>(b * L + i_tok);
        const __nv_bfloat16* ptr_k = qkv + tok_global * paso_qkv + (hq + h) * 64;
        float par[16], impar[16];
        rms_y_reloj_intercalado(ptr_k, seq + i_tok, 1.0f, rope_theta, mitad, par, impar);
        const int64_t loc = cache_loc[tok_global];
        uint8_t* k_dst = kc + loc * BYTES_FILA_CACHE + h * 64;
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const int idx_par = mitad * 16 + m;
            const uint8_t k8_p = a_e4m3_par(par[m] / ks);
            const uint8_t k8_i = a_e4m3_par(impar[m] / ks);
            if (r == 0) {
                k_dst[2 * idx_par] = k8_p;
                k_dst[2 * idx_par + 1] = k8_i;
            }
            k_nuevas[i_tok][2 * idx_par] = e4m3_a_f32(k8_p);
            k_nuevas[i_tok][2 * idx_par + 1] = e4m3_a_f32(k8_i);
        }
    }

    // Los 64 hilos cargan y cuantizan V para las L piezas nuevas
    for (int i_tok = 0; i_tok < L; ++i_tok) {
        const int64_t tok_global = static_cast<int64_t>(b * L + i_tok);
        const __nv_bfloat16* ptr_v = qkv + tok_global * paso_qkv + (hq + hkv + h) * 64;
        const float vf = __bfloat162float(ptr_v[tid]);
        const uint8_t v8 = a_e4m3_par(vf / vs);
        if (r == 0) {
            const int64_t loc = cache_loc[tok_global];
            vc[loc * BYTES_FILA_CACHE + h * 64 + tid] = v8;
        }
        v_nuevas[i_tok][tid] = e4m3_a_f32(v8);
    }
    __syncthreads();

    // 3. Acoplar las piezas causales i_tok <= j al flujo del Grupo de Renormalizacion
    const float q_d = q_fila[tid];
    __shared__ float dot_parcial[2];
    for (int i_tok = 0; i_tok <= j; ++i_tok) {
        float prod = q_d * k_nuevas[i_tok][tid];
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            prod += __shfl_xor_sync(0xffffffffu, prod, off);
        }
        if ((tid & 31) == 0) dot_parcial[tid >> 5] = prod;
        __syncthreads();
        const float s_nuevo = (dot_parcial[0] + dot_parcial[1]) * escala;
        const float m_nuevo = fmaxf(m_acc, s_nuevo);
        const float gamma_a = exp2_exacto(m_acc - m_nuevo);
        const float p_nuevo = exp2_exacto(s_nuevo - m_nuevo);
        z_acc = z_acc * gamma_a + p_nuevo;
        o_dim = o_dim * gamma_a + p_nuevo * v_nuevas[i_tok][tid];
        m_acc = m_nuevo;
        __syncthreads();
    }

    // 4. Colapso del Observable: Normalizacion RMS + Compuerta Sigmoide
    float o_val = (z_acc > 0.f) ? (o_dim / z_acc * vs) : 0.f;
    float sq = o_val * o_val;
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        sq += __shfl_xor_sync(0xffffffffu, sq, off);
    }
    if ((tid & 31) == 0) dot_parcial[tid >> 5] = sq;
    __syncthreads();
    const float inv_rms = rsqrtf((dot_parcial[0] + dot_parcial[1]) * (1.f / 64.f) + 1e-5f);
    o_val *= inv_rms;

    const int64_t tok_actual = static_cast<int64_t>(b * L + j);
    const float puerta = __bfloat162float(qkv[tok_actual * paso_qkv + (hq + 2 * hkv + cabeza_q) * 64 + tid]);
    const float sig = 1.f / (1.f + __expf(-puerta));
    out[tok_actual * (hq * 64) + cabeza_q * 64 + tid] = __float2bfloat16_rn(o_val * sig);
}

} // namespace relativistic_rg
"""

_CUDA_RG_HOST = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void lanzar_rg_completo(
    torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc,
    torch::Tensor tabla, torch::Tensor kc, torch::Tensor vc,
    torch::Tensor k_scale, torch::Tensor v_scale,
    torch::Tensor m_buf, torch::Tensor z_buf, torch::Tensor o_buf,
    torch::Tensor out,
    int64_t pagina, int64_t chunk, int64_t n_splits,
    int64_t L, int64_t hq, int64_t hkv, int64_t ventana,
    double rope_theta, double qk_escala, double escala_log2) {

    const int smem_bytes = relativistic_rg::ETAPAS_TUBO * relativistic_rg::BYTES_ETAPA;
    static bool configurado = false;
    if (!configurado) {
        cudaFuncSetAttribute(
            relativistic_rg::rg_etapa1_marcos_locales,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        );
        configurado = true;
    }

    const int batch = seq_lens.size(0);
    const int n_bh = batch * static_cast<int>(hkv);
    const int n_filas = static_cast<int>(L) * static_cast<int>(hq / hkv);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 malla1(static_cast<unsigned>(n_splits), static_cast<unsigned>(n_bh));
    relativistic_rg::rg_etapa1_marcos_locales<<<malla1, relativistic_rg::HILOS_SCAN, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), qkv.stride(0),
        seq_lens.data_ptr<int>(),
        tabla.data_ptr<int>(), tabla.stride(0), static_cast<int>(pagina),
        reinterpret_cast<const uint8_t*>(kc.data_ptr()), reinterpret_cast<const uint8_t*>(vc.data_ptr()),
        k_scale.data_ptr<float>(),
        m_buf.data_ptr<float>(), z_buf.data_ptr<float>(), o_buf.data_ptr<float>(),
        static_cast<int>(chunk), static_cast<int>(n_splits),
        static_cast<int>(L), static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(ventana),
        static_cast<float>(rope_theta), static_cast<float>(qk_escala), static_cast<float>(escala_log2)
    );

    dim3 malla2(static_cast<unsigned>(n_bh), static_cast<unsigned>(n_filas));
    relativistic_rg::rg_etapa2_colapso_causal<<<malla2, 64, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), qkv.stride(0),
        seq_lens.data_ptr<int>(), cache_loc.data_ptr<int64_t>(),
        reinterpret_cast<uint8_t*>(kc.data_ptr()), reinterpret_cast<uint8_t*>(vc.data_ptr()),
        k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
        m_buf.data_ptr<float>(), z_buf.data_ptr<float>(), o_buf.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        static_cast<int>(n_splits), static_cast<int>(L), static_cast<int>(hq), static_cast<int>(hkv),
        static_cast<float>(rope_theta), static_cast<float>(qk_escala), static_cast<float>(escala_log2)
    );
}
"""


def _compilar_modulo_rg():
    if "mod" not in _MODULO_RG:
        from torch.utils.cpp_extension import load_inline
        _MODULO_RG["mod"] = load_inline(
            name="relativistic_rg_engine_v1",
            cpp_sources="""
void lanzar_rg_completo(
    torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc,
    torch::Tensor tabla, torch::Tensor kc, torch::Tensor vc,
    torch::Tensor k_scale, torch::Tensor v_scale,
    torch::Tensor m_buf, torch::Tensor z_buf, torch::Tensor o_buf,
    torch::Tensor out,
    int64_t pagina, int64_t chunk, int64_t n_splits,
    int64_t L, int64_t hq, int64_t hkv, int64_t ventana,
    double rope_theta, double qk_escala, double escala_log2);
""",
            cuda_sources=_CUDA_RELATIVISTIC_RG + _CUDA_RG_HOST,
            functions=["lanzar_rg_completo"],
            extra_cuda_cflags=["-O3", "-std=c++20", "--use_fast_math"],
            verbose=False,
        )
    return _MODULO_RG["mod"]


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, largo = cfg["batch"], cfg["q_len"]
    hq, hkv, dim, pagina = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    ventana = cfg["window"]
    rope_theta = float(cfg["rope_theta"])
    qk_escala = float(cfg["qk_scale_factor"])
    escala_log2 = float(cfg["sm_scale"]) * LOG2E

    qkv = data["qkv"].reshape(batch * largo, -1)
    tabla = data["page_table"]
    device = qkv.device
    out = torch.empty((batch, largo, hq * 64), dtype=torch.bfloat16, device=device)

    if device.type == "cuda":
        # Particion espaciotemporal adaptativa para saturar los SMs sin atomicas
        max_llaves = (ventana - 1) if ventana > 0 else (tabla.shape[1] * pagina)
        n_bh = batch * hkv
        objetivo_bloques = 132 * 4  # ~4 bloques por SM en H200
        n_splits = max(1, min(64, (objetivo_bloques + n_bh - 1) // n_bh))
        chunk = max(64, ((max_llaves + n_splits - 1) // n_splits + 63) // 64 * 64)
        n_splits = max(1, (max_llaves + chunk - 1) // chunk)

        n_filas = largo * (hq // hkv)
        m_buf, z_buf, o_buf = _buffers_rg(device, n_bh, n_splits, n_filas)

        mod = _compilar_modulo_rg()
        mod.lanzar_rg_completo(
            qkv, data["seq_lens"], data["cache_loc"], tabla,
            data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
            m_buf, z_buf, o_buf, out,
            pagina, chunk, n_splits, largo, hq, hkv, ventana,
            rope_theta, qk_escala, escala_log2
        )
        return out

    from submission_v15 import _metodo_pedazos
    return _metodo_pedazos(data)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for L_test, win_test, theta_test in [(1, 0, 0.0), (1, 2048, 500000.0), (8, 2048, 500000.0)]:
        B, Hq, Hkv, D = 4, 16, 2, 64
        max_pages, page_size = 512, 1
        cfg: Config = {
            "batch": B, "q_len": L_test, "num_q_heads": Hq, "num_kv_heads": Hkv,
            "head_dim": D, "page_size": page_size, "window": win_test, "rope_theta": theta_test,
            "qk_scale_factor": 1.0, "sm_scale": 0.125, "kv_scale": 1.0,
        }
        data: Inputs = {
            "qkv": torch.randn((B, L_test, (2 * Hq + 2 * Hkv) * D), dtype=torch.bfloat16, device=device),
            "positions": torch.arange(B * L_test, dtype=torch.int64, device=device),
            "cache_loc": torch.arange(B * L_test, dtype=torch.int64, device=device),
            "cache_seqlens": torch.full((B,), 256 + L_test, dtype=torch.int32, device=device),
            "cu_seqlens_q": torch.arange(0, (B + 1) * L_test, L_test, dtype=torch.int32, device=device),
            "scheduler_metadata": None,
            "k_cache": torch.zeros((B * max_pages, page_size, Hkv, D), dtype=torch.float8_e4m3fn, device=device),
            "v_cache": torch.zeros((B * max_pages, page_size, Hkv, D), dtype=torch.float8_e4m3fn, device=device),
            "k_scale": torch.tensor(1.0, dtype=torch.float32, device=device),
            "v_scale": torch.tensor(1.0, dtype=torch.float32, device=device),
            "page_table": torch.arange(B * max_pages, dtype=torch.int32, device=device).reshape(B, max_pages),
            "seq_lens": torch.full((B,), 256, dtype=torch.int32, device=device),
            "config": cfg,
        }
        res = kernel_fn(data)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"RelativisticRG OK | L={L_test} | win={win_test} | shape={res.shape} | NaNs={res.isnan().sum().item()}")