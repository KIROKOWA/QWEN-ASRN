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
python muse_attention_decode/compilar_sm90.py                          # ¿compila para la H200? (sin GPU)
python muse_attention_decode/medir.py oficial/                         # µs por forma, con CUDA graphs y L2 frío
okc check muse_attention_decode/submission.py                          # el juez, fuera del tablero
okc submit muse_attention_decode/submission.py
```

`okc login` lo tiene que correr una persona: acepta las reglas y los términos. El reto solo publica `okc` para macOS
y Linux; en Windows (o en Colab) usa [`okc_win.py`](okc_win.py), un cliente de su API HTTP que solo necesita Python:
`python okc_win.py login`, `check submission.py`, `submit submission.py`, `status <job>`, `leaderboard`. Al enviar, el código queda bajo
Apache 2.0, la misma licencia de este repositorio.

### Versiones

| archivo | qué tiene | juez (H200) |
|---|---|---|
| v1 (historia de git) | dos kernels, ~2 programas por SM | 0.87× |
| `submission_v2.py` | ~16 programas por SM, bloques de 128, tabla de páginas pedida antes | 1.19× (la mejor) |
| `submission.py` (v3) | un programa por par de cabezas kv (filas del caché completas); en verify solo 1 programa por SM | 0.83× |
| `submission_v4.py` | v3 + planificador por longitudes reales + doble búfer en registros, evict_first | por medir |
| `submission_v5.py` | v2 + planificador + anillo asíncrono de 3–4 etapas en memoria compartida, configuración por forma medida con ptxas | por medir |
| `submission_v6.py` | **un solo kernel**: softmax con energía de referencia fija (Cauchy–Schwarz), acumulación atómica, cierre por quórum, prefetch al L2, consultas FP8 en verify | 0.98× (mejor en ventana, peor en contexto completo) |
| `submission_v7.py` | v6 con el reparto de la v2 (~16 programas por SM), sin `evict_first` ni prefetch | 1.16× |
| `submission_v8.py` | híbrida: en cada forma, el método que ganó en el juez (quórum con pocas peticiones y en verify con ventana y muchas; pedazos de la v2 en el resto) | **1.27×** |
| `submission_v9.py` | v8 + experimentos: verify con contexto completo con quórum y ranuras (sin atómicos), decode completo con anillo de 3 etapas, decode con ventana con la configuración de la v6 | 0.91× (las ranuras: verify 1 completo 190 µs) |
| `submission_v10.py` | v8 + solo lo que ganó en la v9: decode con ventana con la configuración de la v6 (6.0 µs) | **1.287×** (medida como la v11 sin CUDA) |
| `submission_v11.py` | v10 + **CUDA nativo** en decode con contexto completo y 5+ peticiones ([`cuda_completo.cu`](muse_attention_decode/cuda_completo.cu)): las dos cabezas kv por bloque (filas de 128 bytes), anillo `cp.async` de 4 etapas con swizzle, `mma.sync` en tensor cores, energía de referencia fija; si no compila, vuelve a Triton | 1.287× por Triton (el CUDA no compiló: `-std=c++17`, el PyTorch del juez exige C++20); es la v10 medida |
| `submission_v12.py` | v11 compilando con `-std=c++20` | 1.19×: el CUDA **compiló y es correcto**, pero más lento (186/428 µs contra 156/306 de Triton): 219 registros, 8 warps por SM |
| `submission_v13.py` | CUDA con 8 warps por bloque (uno por cabeza kv y grupo de llaves), `p·v` traspuesto (acumuladores sin relleno), `q` desde memoria compartida, direcciones recalculadas por vuelta: **80 registros, sin derrames, 3 bloques por SM (24 warps)** | por medir |

`compilar_cuda.py` compila el CUDA de una versión para sm_90a sin GPU (NVRTC + `ptxas`), y `simular_cuda.py` simula el kernel carril por carril en numpy (fragmentos de `mma.sync`, swizzle, anillo) contra un cálculo directo.

### v6: la atención como función de partición

Ideas de otras áreas, cada una porque resuelve algo concreto de este problema:

- **Física estadística: energía de referencia fija.** El softmax es una función de partición, Z = Σ e^(s_i), y cada
  estado se puede medir contra cualquier energía de referencia c sin cambiar el resultado. FlashAttention usa el máximo
  corriendo, que obliga a calcular el máximo de cada bloque y a reescalar lo acumulado. Muse normaliza consultas y
  llaves (QK-norm), así que por **Cauchy–Schwarz** |q·k| ≤ |q||k| = (5.473·8)·8: cada logit (en base 2) cae en un
  intervalo de ancho ~126, que cabe en el rango normal de float32. Con c = techo − 40, todos los términos quedan en
  [2^-86, 2^40]: sin máximo, sin reescalar, sin desbordes.
- **Cantidades extensivas.** Con la misma c en todos los programas, Z y Σ e^(s−c)·v se suman como la masa o la
  energía: cada programa los agrega con atómicos a un acumulador global. Ya no hay pedazos que guardar ni juntar (en
  la v2, en verify, los pedazos movían 4 veces más bytes que el propio caché).
- **Biología: quorum sensing.** Cada programa, al terminar, suma 1 a un contador de su grupo; el que completa el
  quórum (el último en llegar) cierra: divide, normaliza, aplica la puerta y escribe, y deja acumuladores y contador en
  cero para la siguiente llamada. Un solo lanzamiento de kernel (SGLang usa cinco).
- **Bases de datos: prefetching de sondeos sueltos** (como AMAC o el prefetching por grupos en hash joins): las filas
  de vueltas futuras se piden al L2 con `prefetch.global.L2`, sin ocupar registros ni memoria compartida.
- **Teoría de colas** (v4) y **anillo asíncrono** (v5), y los programas de un mismo trabajo (sus dos cabezas kv) son
  vecinos en la malla: piden las mismas filas de 128 bytes casi a la vez, y la memoria las trae una vez.

`python muse_attention_decode/compilar_sm90.py --archivo submission_v4.py` compila para la H200 sin GPU y reporta
registros, derrames, memoria compartida y programas por SM de cada kernel.

### Qué falta probar para que sea más rápido

1. Afinar en la H200 `BLOQUE_N`, `num_warps`, `num_stages` y el número de pedazos (ahora apunta a unos dos programas
   por SM).
2. Leer K y V con TMA o con cargas de 16 bytes. Con páginas de 1, cada pieza es una fila suelta de 128 bytes por cabeza.
3. En las capas de contexto completo con 16–32 peticiones, el piso es de 63–127 µs, dominado por leer el caché; ahí
   la meta es acercarse al ancho de banda, no ahorrar lanzamientos.
