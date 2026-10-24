# ASRN en el Open Kernel Challenge

Entradas para el [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/) (Paradigm, Meta y Andromeda):
hacer más rápidos los kernels de Muse Spark Open en una H200, contra los de SGLang. Las ideas vienen de [ASRN](https://github.com/KIROKOWA/ASRN); el
modelo no cambia (el reto exige atención exacta sobre todas las llaves, así que los casilleros de hash de ASRN no
entran tal cual).

## `muse_attention_decode`: atención con puerta en dos kernels

SGLang lanza cinco kernels por paso (FusedQKNorm, store_cache, la consulta a FP8, FA3 y rms_norm_gate). En las capas
de ventana deslizante, que pesan 46 de cada 62 llamadas, el trabajo es chico (unos 3 µs de datos para 32 peticiones) y
lo que más cuesta son los lanzamientos. [`submission.py`](muse_attention_decode/submission.py) lo hace en dos:

| kernel | programas | qué hace |
|---|---|---|
| `_parcial` | (pedazo del contexto, petición, cabeza kv) | normaliza y rota sus consultas, lee las llaves y valores FP8 ya guardados de su pedazo y deja (máximo, suma, acumulado) del softmax en línea |
| `_final` | (petición, cabeza kv, fila de consulta) | junta los pedazos, agrega las piezas nuevas del bloque (que normaliza, rota, redondea a FP8 y guarda en el caché) y aplica rmsnorm × sigmoid(gate) |

Lo que viene de ASRN:

- **Puntajes que nunca salen de la SRAM**, como la atención por casilleros fusionada de [`asrn/nucleo.py`](https://github.com/KIROKOWA/ASRN/blob/main/asrn/nucleo.py).
- **La unión de pedazos es una composición asociativa**, la misma estructura que `_componer` en el escaneo paralelo
  de ASRN: `(m1, l1, a1) ∘ (m2, l2, a2) = (m, l1·2^(m1−m) + l2·2^(m2−m), a1·2^(m1−m) + a2·2^(m2−m))`.
- **Filas sin llaves válidas con un máximo finito**, como `_softmax_valido` en [`asrn/estable.py`](https://github.com/KIROKOWA/ASRN/blob/main/asrn/estable.py): nunca dan NaN.

`_parcial` solo lee posiciones anteriores al bloque y `_final` solo escribe las del bloque, así que no hay carrera en
el caché. No sincroniza con la CPU ni ramifica sobre valores, así que se puede capturar en un CUDA graph.

### Estado

- Correcto contra el `reference.py` del reto en 11 casos parecidos a las pruebas del juez: ventana y contexto
  completo, verificación de 8 piezas, páginas de 1 y 64, escala 1/64, `edges`, `soft`, `coherent` y cabezas `dp`.
  El error L2 relativo es de alrededor del 1%, contra un límite de 13–15%. Se probó en CPU con el intérprete de
  Triton (`TRITON_INTERPRET=1`).
- **Todavía no se ha corrido en una GPU.** La velocidad y la compilación en Hopper se comprueban con el
  [cuaderno de Colab](muse_attention_decode/colab.ipynb) (L4 o H100) y luego con `okc check`.

### Uso

```bash
okc init muse_attention_decode oficial/        # reference.py, checks.py, baseline.py, TASK.md (no se suben al repo)
python muse_attention_decode/probar_local.py oficial/ [--grandes]     # ¿correcto? (GPU; en CPU: TRITON_INTERPRET=1)
python muse_attention_decode/medir.py oficial/                         # µs por forma, con CUDA graphs y L2 frío
okc check muse_attention_decode/submission.py                          # el juez, fuera del tablero
okc submit muse_attention_decode/submission.py
```

`okc login` lo tiene que correr una persona: acepta las reglas y los términos. El reto solo publica `okc` para macOS
y Linux; en Windows (o en Colab) usa [`okc_win.py`](okc_win.py), un cliente de su API HTTP que solo necesita Python:
`python okc_win.py login`, `check submission.py`, `submit submission.py`, `status <job>`, `leaderboard`. Al enviar, el código queda bajo
Apache 2.0, la misma licencia de este repositorio.

### Qué falta probar para que sea más rápido

1. Afinar en la H200 `BLOQUE_N`, `num_warps`, `num_stages` y el número de pedazos (ahora apunta a unos dos programas
   por SM).
2. Leer K y V con TMA o con cargas de 16 bytes. Con páginas de 1, cada pieza es una fila suelta de 128 bytes por cabeza.
3. En las capas de contexto completo con 16–32 peticiones, el piso es de 63–127 µs, dominado por leer el caché; ahí
   la meta es acercarse al ancho de banda, no ahorrar lanzamientos.
