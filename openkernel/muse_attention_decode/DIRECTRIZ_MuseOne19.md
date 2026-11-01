# Directriz maestra de MuseOne19 contra lo que MuseOne17 ya hace

Fuente: `prompt_maestro_muse_one19.md` (FlashAttention-3, FlashMLA, FlashInfer, CUTLASS, Colfax, reglas del juez).
Evidencia: el código de `MuseOne17.py` y las fases medidas en el juez con `MuseOne17diag.py` (check
`46c38b7ea5c74e54`).

## Ya está en MuseOne17

| punto de la directriz | en MuseOne17 |
|---|---|
| `wgmma.mma_async` FP8, acumulación fp32 | QK^T en wgmma E4M3 directo del anillo; en verify también PV en FP8 (P en E4M3, como FA3) |
| TMA / `cp.async.bulk` + `mbarrier` | índices de página por `cp.async.bulk` con mbarrier; anillo K/V de 3–4 etapas con `cp.async.cg` (filas sueltas de 128 B con página de 1: TMA tensorial no aplica) |
| especialización productor/consumidor + `setmaxnreg` | kernel plano (capas completas B > 1): un warpgroup productor (`setmaxnreg.dec`) y consumidores (`setmaxnreg.inc`) |
| `ldmatrix` / `stmatrix`, `prmt.b32` para V en FP8 | V^T con `prmt` + `cvt.rn.f16x2.e4m3x2`, `ldmatrix.trans`, `stmatrix` |
| empaquetado GQA + tokens de verify (FlashInfer) | baldosa de 64 filas = 8 tokens × 8 cabezas por cabeza kv: cada llave se lee una vez |
| `prefetch.global.L2` | tabla de páginas del primer segmento y filas de la puerta |
| base 2 (`ex2.approx`), sigmoide con `rcp.approx` | `ex2.approx.ftz`; `sigm` = `__fdividef(1, 1 + __expf(-x))` (ex2 + rcp aproximados) |
| swizzle de 128 B | anillo K/V con XOR de 128 B (descriptores SW128 de wgmma) |
| un solo lanzamiento, split-K con último CTA, contadores que se reinician solos | sí, más un cierre cooperativo (todos los CTAs cierran una parte) donde la malla cabe entera |
| despachador por forma (1/16/32, decode/verify) | `_plan` + `_TUNED`, medidos por vjeux en H200 |
| acumuladores y estadísticas en fp32 | sí (pasa las 16 pruebas con margen) |
| `-arch=sm_90a`, sin asignaciones ni sincronizaciones en la captura | sí |

## No conviene

- `__threadfence()` en el cierre: es `fence.sc.gpu` en todos los hilos. MuseOne17 usa `bar.sync` + un
  `atom.acq_rel.gpu` del hilo 0, que es más barato (MuseOne10 lo midió a favor).
- `ld.global.cs` / evict_first en K/V: el L2 empieza frío y cada fila se lee una sola vez, así que no hay nada
  que proteger; en MuseOne v7 evict_first fue más lento.
- `--use_fast_math` y `-Xptxas=-v`: el kernel ya usa intrínsecos aproximados donde importa; fast-math solo
  agrega riesgo de precisión en las normas, y `-v` llena el log del juez.
- Ping-pong entre warpgroups en verify: vjeux lo midió (OKC_TURN) y fue 1.3 % más lento.
- Repartos dinámicos con más segmentos por CTA: cada frontera de segmento cuesta ~7 µs en verify (recarga de q,
  vaciado del anillo, merge); vjeux ya corta los repartos en un espacio de llaves virtual para evitarlas.

## Lo que sí falta (según las fases medidas)

1. Cierre entre CTAs en las formas chicas: 1.5–3 µs (40–45 % de la ventana con 1 petición). Opción sin L2: clusters
   con memoria compartida distribuida; MuseOne18a lo prueba en decode 1 ventana.
2. Desbalance en las capas completas grandes: 3.7 / 6 µs entre la mediana y el último CTA en terminar.
3. Verify completa: 12–20 µs más que decode con los mismos bytes.

Compilar en el import (punto 5D) no cambia el puntaje: la compilación no cuenta en el tiempo medido.
