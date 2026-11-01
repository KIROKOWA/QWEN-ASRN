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
- MuseOne18a (check `897eea3159f84f2c`, 2.407×): las 6 alternativas perdieron o empataron (kernel, µs):
  d1s clusters de 8: 8.9 contra 4.5–4.7 (**clusters descartados**); d1f 132 CTAs: 11.6 contra 9.7; d16s
  cooperativo: 7.9 contra 6.8–7.0; d32s 11 pedazos: 9.0 (igual); v1s 2 copias: 6.1 (igual); v16s 16 pedazos:
  11.5 contra 10.7. Los planes de vjeux quedan como están.
- Descartado sin gastar checks: `ex2.approx.f16x2` (en sm_90 MUFU es escalar: no da 2×); exponenciales por
  polinomio en FMA (vjeux, OKC_EXPPOLY: más lento, el bucle de verify está limitado por emisión).
- MuseOne19diag (`hacer_diag2.py`): por CTA, SM, segmentos, vueltas, entrada y suma de bucles: tiempo por vuelta
  decode contra verify y quiénes son los CTAs lentos.
- MuseOne19diag (check `bb21701fa7204838`, 2.611×):
  - d32f: vuelta 1.02–1.06 µs en todos los CTAs (4.2 TB/s: techo de lectura). Cola de ~5 µs: CTAs con frontera
    de petición +1.5 µs (SEGK corto), 2º CTA del SM +1.6 µs, 152.25 vueltas promedio → algunos hacen 153.
  - v1s: vuelta 0.86 µs, pero emitir el merge (2048 red.v4 por CTA, 33 CTAs) tarda 0.9–1.0 µs más; 2 copias no
    cambiaron nada (18a): el límite es el total de atómicos del L2.
- MuseOne19: verify con ventana (cierre cooperativo) con una copia privada por split escrita con stores
  (`st.global.cg.v4`) en vez de `red.v4`; el cierre reparte las copias de cada fila entre G grupos de hilos
  (v1s: 33 copias, G = 8, ≤ 2 viajes al L2; v16s: 8 copias, G = 2; v32s: 4 copias, camino de antes).
- diag2 de verify completa (mismo check): la vuelta cuesta lo mismo que en decode (1.05–1.18 contra 1.02–1.06 µs).
  El exceso de verify es el 2º CTA de cada SM: termina 8.5 µs (v16f) y 10–16 µs (v32f) después del 1º (en decode,
  1.6 µs), y la barrera global espera al más lento. Los CTAs lentos son siempre los de blockIdx ≥ 132.
- MuseOne19 también: en verify completa (plano q_len 8, un par kv) la primera mitad de la malla toma W0_L8 = 1135/2048
  de cada par de partes (vjeux barrió 1130–1140 con registro por SM; aquí sin registro, por la mitad de la malla).
- **MuseOne19 (check `2f86cb74dad94696`, 2.553×): los dos cambios perdieron** (kernel, µs, contra MuseOne17):
  - copias privadas: v1s 8.1 contra 6.1–6.5, v16s 11.5 contra 10.7–11.0, v32s 15.1 contra 14.0–14.5. Leer y poner
    en cero 33 copias (1 MB de lecturas + 1 MB de ceros) cuesta más que los atómicos que ahorra.
  - reparto 1135/2048: v16f 105.8 contra 100–101, v32f 195.9 contra 184–186. O el par de CTAs de cada SM no es
    (c, c + 132) en todos los SMs, o el SM ya está ocupado igual con partes iguales; sin registro no sirve.
  - controles: d16f 83.2 y d32f 158.5 (algo mejores que antes: otra máquina / ruido).
  - MuseOne17 sigue siendo la mejor versión.

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
