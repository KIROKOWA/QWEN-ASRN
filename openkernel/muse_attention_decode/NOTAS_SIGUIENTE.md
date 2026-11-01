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

- MuseOne17diag (check `46c38b7ea5c74e54`, 2.597×): fases por CTA (µs desde el primer CTA; mediana/máximo):

  | forma | índices | prólogo | fin del bucle | merge emitido | último CTA | salida |
  |---|---|---|---|---|---|---|
  | d1s | 0.6 | 1.1 | 1.6/2.1 | 1.8/2.3 | 3.2 | 3.8 |
  | d1f | 0.9 | 1.9 | 5.8/6.1 | 6.1/6.4 | 8.1 | 9.1 |
  | d16s | 0.9 | 2.0 | 3.5/4.2 | 3.8/4.5 | 5.4 | 6.2 |
  | d32s | 0.9 | 2.2 | 5.4/6.2 | 5.7/6.5 | 7.2 | 8.1 |
  | v1s | 0.7 | 1.6 | 2.5/3.0 | 3.3/3.8 | 4.8 | 5.5 |
  | d16f | | | 81.1/84.8 | | 83.6 | 86.6 |
  | d32f | | | 156.2/162.3 | | 159.7 | 164.2 |

  - El merge entre CTAs + el cierre cuestan 1.5–3 µs en todas las formas chicas (40–45 % de la ventana con B = 1).
  - Capas completas grandes: 3.7 / 6 µs entre la mediana y el último CTA en terminar el bucle (desbalance).
  - Falta el resto del log (v1f, v16s, v16f, v32s, v32f): se cortó en okc_win.py (arreglado).
- MuseOne18a: un plan alternativo en 6 formas (ver `_EXPERIMENTO`), las otras 6 de control.

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
