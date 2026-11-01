# Notas para la siguiente versión (MuseOne18+)

## Estado

- MuseOne17 (base de vjeux) en el juez, cuenta KIROKOWA: **2.585×** [2.547, 2.637], check `f9482312e3324333`.
  Tiempo de GPU del kernel por forma (µs; las celdas con ≈ venían truncadas en el log):

  | forma | ventana | completa |
  |---|---|---|
  | decode 1 | 4.7 | 9.8 |
  | decode 16 | 6.8 | 88.0 |
  | decode 32 | 8.7 | 164.6 |
  | verify 1 | 6.1 | 13.6 |
  | verify 16 | 10.7 | ≈100 |
  | verify 32 | ≈14 | ≈185 |

- Pendiente: `okc check MuseOne17diag.py` (fases por CTA) para elegir el primer cambio.

## Prioridades (de ANALISIS_MuseOne17.md)

1. Verify completa B16/B32: 12–20 µs más que decode con los mismos bytes (cadena QK → softmax → PV de los
   consumidores; superponer el QK del bloque i+1 con el softmax del bloque i).
2. B = 1 (ventana 4.7 / 6.1 µs, completa 9.8 / 13.6 µs): latencia del merge entre CTAs y de la cadena seq → tabla →
   K/V con el L2 frío.

## Referencia anotada: "Accelerating MoE model inference with Locality-Aware Kernel Design"

https://pytorch.org/blog/accelerating-moe-model/ (Hoque, Wright, Virós Martin, Yang; PyTorch, abril 2024).
Continuación: https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/
(agosto 2025).

- Qué hacen: el kernel Fused MoE de vLLM (Triton) tenía poca tasa de aciertos en el L2. Cambiar el orden en que los
  programas recorren los bloques (column-major en vez de row-major, o "grouped launch") hace que programas
  vecinos en el tiempo lean los mismos pesos mientras siguen en el L2: hasta ~4× en A100 y ~4.4× en H100. La
  continuación: kernel persistente con orden agrupado para reutilizar A y B en el L2 (DeepSeek-V3, hasta 2.62×).
- Para esta atención aplica poco: cada llave se lee una vez por llamada y el L2 empieza frío, así que no hay
  reuso entre CTAs que ordenar (las dos cabezas kv ya comparten cada fila de 128 B dentro del mismo CTA).
  Donde sí puede servir:
  - el orden de despacho de los CTAs en las capas completas planas (los dos CTAs de un SM y los vecinos en el
    tiempo leen páginas de la misma petición: las filas de la tabla de páginas sí se reusan en el L2);
  - los acumuladores del merge (atómicos al L2): agrupar los CTAs de una misma petición en el tiempo para que su
    cierre lea líneas calientes.
- Aplica de lleno a los otros problemas del reto (`muse_moe_w4a16`, `muse_decode_layer`, `muse_prefill_layer`):
  grouped GEMM de expertos NVFP4, donde el orden de los tiles decide el reuso de pesos en el L2.
