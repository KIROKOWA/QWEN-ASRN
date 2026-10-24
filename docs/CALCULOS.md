# Qwen 2.5 7B + ASRN "1 y 1": cálculos de VRAM y FLOPs (A100 80 GB)

Todo sale de `python -m qwen_asrn.presupuesto` (Python puro): cambia `--largo`, `--lote`, `--ventana` o
`--sumideros` y lo recalcula. Hay tres tipos de número:

- **Exacto**: los conteos de parámetros. Las pruebas los comparan con los modelos reales; 7 615 616 512 es el total
  oficial de Qwen 2.5 7B.
- **Contado**: las activaciones. Es lo que autograd guarda con las funciones de `qwen_asrn/modelo.py`. En un Qwen2
  diminuto la fórmula difiere de lo que guarda autograd en un +1.0 %, y esa diferencia es cos/sin de RoPE.
  `tests/test_qwen_asrn.py::test_presupuesto_cuenta_lo_que_guarda_autograd` lo comprueba.
- **Estimado**: los picos transitorios, el contexto de CUDA y la velocidad. Todavía no se midieron en una A100.

Configuración: Qwen 2.5 7B (28 bloques, d = 3584, 28/4 cabezas, ff = 18 944, V = 152 064, salida sin atar). Ventana
W = 1024 con S = 4 sumideros. `CapaASRNEstable` usa las opciones por defecto de `asrn` (T = 8 tablas, 14 bits,
chunk 32, confirmar 16) y el grupo A de protecciones.

## 1. A o B: se elige A

|  | A: atención de Qwen con ventana + ASRN cada 2 bloques | B: 14 bloques impares con conv k=4 en vez de atención |
|---|---|---|
| Arranque neutro exacto | **sí** (W_o = 0; idéntico a Qwen si L ≤ W) | **imposible**: quitar 14 atenciones cambia las activaciones en el paso 0, sin importar cómo se inicie la conv |
| FLOPs/pieza respecto de A | — | conv densa (`BloqueLocal`): **+0.204 GFLOP**; conv por canal: −1.234 GFLOP (−7.8 %) |
| Parámetros nuevos | 0 | conv densa: +719 373 312 (10.72 GiB con AdamW fp32); por canal: +250 880 |
| Pesos de Qwen que se liberan | — | 0.77 GiB |
| Caché KV que se ahorra por texto | — | 28.2 MiB (de 56.4) |

Con una ventana corta, la atención de Qwen cuesta poco: 0.825 GFLOP de 15.76 por pieza. La convolución densa de
`BloqueLocal` (d × d × 4) cuesta más que las proyecciones q/k/v/o que reemplaza. La convolución por canal sí ahorra
un 7.8 %, pero a cambio se pierden 14 capas de atención preentrenadas que el MLP congelado de cada bloque espera
recibir. Además, B incumple el candado de arranque neutro. **Se elige A.**

Una salvedad sobre A: Qwen 2.5 se entrenó con atención completa. Con L ≤ W el modelo de partida es exactamente Qwen;
con L > W es Qwen con ventana. Los S = 4 sumideros (StreamingLLM) evitan que se degrade cuando el texto pasa de la
ventana. Las capas ASRN quedan después de los bloques 0, 2, …, 26: son 14 capas, como en `ModeloASRN` con
`largo_cada=2`.

## 2. Parámetros (exactos)

| | parámetros | bytes |
|---|---|---|
| Qwen 2.5 7B (bf16, congelado) | 7 615 616 512 | 15 231 233 024 (14.19 GiB) |
| una `CapaASRNEstable(3584)` | 26 518 306 (v 12 845 056 + W_o 12 845 056 + hash 695 536 + resto 132 658) | 106 073 224 fp32 |
| 14 capas ASRN (fp32, entrenables) | 371 256 284 | 1 485 025 136 (1.38 GiB) |
| gradientes ASRN fp32 | | 1.38 GiB |
| AdamW m y v fp32 (solo ASRN) | | 2.77 GiB |

## 3. Activaciones con `checkpoint='asrn'` (contadas)

Los pesos de Qwen no tienen gradiente, así que sus `Linear` guardan solo el peso y nada por pieza. Aun así, el
gradiente tiene que atravesar los bloques 1 a 27 para llegar a las capas ASRN anteriores, y cada uno de esos 27
bloques guarda lo que necesita para calcular dX. El bloque 0 no guarda nada, porque va antes de la primera capa ASRN.
Bytes por pieza y por bloque:

| tensor guardado | bytes/pieza |
|---|---|
| entradas de las 2 RMSNorm (bf16; `_RMSNorm` no guarda la copia fp32) | 14 336 |
| q (SDPA) | 7 168 |
| k, v por bloques (2W + 8 llaves por cada W preguntas, sin repetir las cabezas) | 4 112 |
| salida de la atención | 7 168 |
| logsumexp fp32 | 112 |
| gate y up del SwiGLU (bf16; `_SwiGLU` no guarda silu(gate)) | 75 776 |
| **bloque** | **108 672** × 27 = 2 934 144 |
| ASRN: entrada h de cada capa (bf16) × 14 + E una vez + x | 107 528 |
| norma final + etiquetas | 7 176 |
| máscara aditiva bf16, **una para todas las capas** | 28 784 |
| **total** | **3 077 632 B = 2.935 MiB por pieza** |

La máscara se construye una sola vez como máscara aditiva bf16 (0 / −inf). Si se le pasa a SDPA una máscara booleana,
SDPA arma su propia copia aditiva en cada capa y cada capa la guarda. Eso medía +29 KB por pieza y por capa, o sea
+0.78 MB por pieza en total.

## 4. VRAM total en una A100 80 GB

| | lote × largo 1 × 4096 | 1 × 8192 | 1 × 16 384 |
|---|---|---|---|
| pesos Qwen bf16 | 14.19 GiB | 14.19 | 14.19 |
| ASRN fp32 + gradientes + AdamW | 5.53 GiB | 5.53 | 5.53 |
| activaciones (`checkpoint='asrn'`) | 11.74 GiB | 23.48 | 46.96 |
| pico de una capa ASRN en la vuelta (estimado, ~256 KB/pieza) | 1.00 GiB | 2.00 | 4.00 |
| pérdida por trozos de 1024 piezas (logits de un trozo) | 1.45 GiB | 1.45 | 1.45 |
| contexto CUDA + fragmentación (estimado) | 3.00 GiB | 3.00 | 3.00 |
| **total** | **36.9 GiB** | **49.7 GiB** | **75.1 GiB** (justo) |

- Máximo de piezas por micro-lote: unas **17 950** con `checkpoint='asrn'` y unas **100 000** con
  `checkpoint='todo'` (`todo` recalcula también los bloques de Qwen; con 65 536 piezas ocupa 60.7 GiB).
- **Recomendado: 1 × 8192 (o 2 × 4096), con `--acumular` para agrandar el lote efectivo.**
- Nunca se arman los logits de todo el lote. Con 8192 piezas serían 8192 × 152 064 × (2 + 4 + 4) B = 12.5 GB; por
  trozos son 1.45 GiB.

## 5. FLOPs por pieza

| | GFLOP/pieza |
|---|---|
| ida: Qwen lineal (28 × 2 × 233 046 016 + salida 2 × 544 997 376) | 14.141 |
| ida: atención por bloques (4 × (2W + 8) × 3584 × 28; útil con W llaves: 0.411) | 0.825 |
| ida: ASRN (14 × 56.82 M: v y W_o 51.38 M, búsqueda 3.80 M, hash 1.39 M, proyecciones 0.25 M) | 0.796 |
| **ida** | **15.761** |
| vuelta: Qwen congelado solo dX (bloques 1–27 + salida) + atención ×2.5 + ASRN ×2 (dX y dW) | 17.255 |
| recálculo: ASRN (checkpoint) + salida (pérdida por trozos) | 1.886 |
| **paso de entrenamiento** | **34.90** (con `checkpoint='todo'`: 48.28) |

El paso cuesta 2.2 veces la ida. Un ajuste fino completo costaría unas 3 veces la ida, porque además calcularía dW.
En una A100 (312 TFLOPS bf16) con un MFU del 40 % serían **unas 3 600 piezas/s** con `checkpoint='asrn'` y unas
2 600 con `todo`. Es una estimación: los casilleros de ASRN no son una multiplicación de matrices grande y tienen otro
rendimiento.

## 6. Escritura O(1) acoplada

Por pieza nueva:

- **Qwen**: 1 pregunta contra W + 8 llaves en cada bloque (caché KV en anillo más sumideros).
- **ASRN**: T × chunk = 256 candidatos por capa (anillos de casilleros), Hawkes por clave (`_MemoriaHawkes`) y la
  racha (un número).

| | valor |
|---|---|
| FLOPs/pieza | 15.32 GFLOP (Qwen 14.14, atención 0.41, ASRN 0.77) |
| pesos leídos por paso | 16.72 GB → 8.2 ms a 2.039 TB/s (≤ 122 pasos/s; cada paso escribe una pieza de cada texto del lote) |
| caché KV por texto: 28 × (1024 + 8) × 2 × 4 × 128 × 2 B | **56.4 MiB, fija** |
| anillos por texto: 14 × T × 2^14 × (4 × 32 + 8) B | **238 MiB, fijos** (guardan posiciones, no vectores: no dependen de d) |

Lo que **crece** con el texto es la memoria de cada posición que una pregunta puede leer, como en `asrn._Escritor`:

| por pieza de contexto | `v_factorizado=False` (como `_Escritor`) | **`v_factorizado=True`** (por defecto) |
|---|---|---|
| por capa: phi 448 + ids 32 + Kc 64 + sufijo 16 + reloj fp64 8 + log α 4 (+ V fp32 14 336) | 14 908 B | 572 B |
| 14 capas + texto + firmas + Hawkes (≤ 672 B) | 209 396 B | **8 692 B** |
| contexto de 1 M de piezas | 195 GiB (no cabe) | **8.1 GiB** |

`v_factorizado`: como v es lineal y sin sesgo, Σₖ attₖ · v(E_{sₖ+1}) = v(Σₖ attₖ · E_{sₖ+1}). En vez de guardar V
(d números fp32 por posición y por capa), se guarda el texto, se toman los 256 vectores de pieza de los candidatos
de la tabla de embeddings de Qwen y se aplica v una sola vez. En las pruebas coincide con la ida completa hasta
2.4e-7, igual que la versión sin factorizar.

La memoria de Hawkes lleva además diccionarios de Python en la CPU: unos 100 B por clave nueva, hasta 4 claves por
pieza y por capa.

## 7. Contrato de precisión y arranque neutro (qué hace el código)

- **Qwen**: bf16 y `requires_grad=False`. El estado h entre bloques va en bf16, como en Qwen. `_RMSNorm` y `_SwiGLU`
  hacen las mismas cuentas que Qwen en la ida y guardan menos para la vuelta.
- **ASRN**: pesos en fp32 estricto; `revisar_contrato()` lo exige en cada ida, junto con c_vistos en fp64. La capa
  recibe `E.float()` y `h.float()` (exacto desde bf16) y corre dentro de `torch.autocast(bf16)`.
  `CapaASRNEstable` conserva el reloj en fp64 (`reloj_64`), su proyección en fp32 real (`_ProyeccionIEEE`, sin TF32
  aunque TF32 esté encendido) y la puerta en fp32. La salida vuelve a bf16.
- **Arranque neutro**: `W_o = 0`, y la puerta g en 0 (g = ½). Así h' = h + g · 0 = h en fp32, y bf16 → fp32 → bf16
  es exacto, de modo que el resultado es idéntico bit a bit. `verificar_arranque_neutro()` lo comprueba, y
  `entrenar` lo verifica antes del primer paso.
- **Por qué W_o y no la puerta**: sigmoid nunca da 0 exacto, y forzar la puerta a cerrarse (con un sesgo muy
  negativo) anula su gradiente. Con W_o = 0 y C = c_t, el gradiente dL/dW_o = Σ_t g_t (dL/dh'_t) C^T **no es cero**
  en el paso 0 (es el mismo patrón que B = 0 en LoRA). El resto de la capa empieza a recibir gradiente desde el
  paso 1; lo verifica `test_solo_asrn_recibe_gradiente_y_W_o_aprende_en_el_paso_0`.
- La puerta g arranca en 0, y no con el valor al azar de `nn.Linear`, porque los estados de Qwen tienen canales de
  magnitud enorme y saturarían la sigmoide desde el inicio.
