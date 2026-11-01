# MuseOne17: de 1.67× a 2.65× y qué falta para 3.34×

## Cómo puntúa el juez (TASK.md oficial)

- 6 cargas: decode 1/16/32 y verify 1/16/32. **Tiempo de una carga = 46 × capa con ventana + 16 × capa completa.**
- Puntaje = **media geométrica** de (SGLang / nuestro) sobre las 6 cargas: cada carga pesa lo mismo.
- El L2 está frío en cada repetición y se restan ~4 µs fijos de la repetición del grafo a ambos lados.

## Tiempos por forma (µs, juez H200)

SGLang y la base de vjeux salen de perfiles públicos del juez (alygg77 job `1c8d81a0dbd94ee0` 2.653×, ruwwww job
`2827c6e61a784868` 2.603× con el mismo archivo). Piso: mover los bytes a 4.8 TB/s.

| forma | SGLang | vjeux (= MuseOne17) | piso | parte de la carga que es ventana |
|---|---|---|---|---|
| decode 1 ventana / completa | 20.75 / 31.65 | 5.19 / 10.95 | 0.11 / 4.33 | 58 % |
| decode 16 ventana / completa | 27.15 / 202.02 | 7.91 / 83.65 | 1.65 / 63.18 | 21 % |
| decode 32 ventana / completa | 31.81 / 387.33 | 9.70 / 159.93 | 3.29 / 126.35 | 15 % |
| verify 1 ventana / completa | 23.23 / 36.58 | 6.77 / 14.25 | 0.12 / 4.28 | 58 % |
| verify 16 ventana / completa | 29.90 / 206.98 | 11.77 / 99.98 | 1.86 / 63.39 | 25 % |
| verify 32 ventana / completa | 34.94 / 392.46 | 14.82 / 182.13 | 3.72 / 126.78 | 19 % |

Por carga (d1, d16, d32, v1, v16, v32): **3.53, 2.63, 2.55, 3.07, 2.19, 2.19** → media geométrica **2.653**.

MuseOne14/16 (1.67–1.69×) estaba detrás en todas: p. ej. decode 32 completa ~260 µs contra 160, y la ventana con
1 petición 6.6 µs contra 5.2. Por eso MuseOne17 parte de la base de vjeux y no de MuseOne16.

## De dónde tendría que salir 3.34× (modelo sobre la base)

| cambio | puntaje |
|---|---|
| base (vjeux) | 2.65 |
| ventana con B = 1 (d1s, v1s) 2 µs más rápida | 2.85 |
| completa con B = 1 (d1f, v1f) 4 µs más rápida | 2.79 |
| verify completa tan rápida como decode completa (v16f 100→84, v32f 182→160) | 2.81 |
| todas las formas con ventana 2 µs más rápidas | 2.93 |
| las tres primeras + ventana −2 µs en todas | 3.25 |
| eso + completas B16/B32 a 4.4 TB/s | 3.52 |

Conclusiones:

1. **El primer lugar no se gana en las capas completas grandes**: decode 16/32 completa ya va a ~4 TB/s (el techo
   medido de lecturas aleatorias de filas de 128 B en la H200 es ~3.9–4.2 TB/s). Ahí queda poco.
2. **Se gana en latencia**: las cargas con B = 1 son 58 % ventana, y la ventana es pura latencia (0.5 MB de datos).
   Cada µs menos en d1s/v1s vale ~+3.5 % de puntaje. La cadena mínima con L2 frío es seq → tabla de páginas → K/V
   (vjeux ya especula la longitud con B = 1 para saltarse el primer viaje) + la mezcla entre CTAs (atómicos + último
   CTA) + la escritura.
3. **Verify completa** (v16f, v32f) está 16–22 µs por encima de decode completa con los mismos bytes: el bucle de
   los consumidores (QK wgmma → softmax → PV) es una cadena dependiente; superponer QK del bloque i+1 con el softmax
   del bloque i (ping-pong entre warpgroups, como FA3) es la palanca conocida.

## Herramientas

- `compilar_nvrtc.py --archivo MuseOne17.py`: compila sin GPU todas las instancias de `attn_kernel` para sm_90a
  (NVRTC + ptxas) e informa registros y derrames. Hoy: 80–128 registros, sin derrames.
- Cualquier cambio en el CUDA se debe medir con `okc check` (20 por día): no hay forma de medir sin la H200.
