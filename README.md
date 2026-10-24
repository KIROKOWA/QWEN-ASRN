# qwen-asrn

Cirugía híbrida "1 y 1" sobre **Qwen 2.5 7B**. Qwen queda congelado en bf16, con atención de ventana W = 1024 y
4 sumideros. Después de los bloques 0, 2, …, 26 se injertan 14 capas
[`asrn.CapaASRNEstable`](https://github.com/KIROKOWA/ASRN) en fp32 con el grupo A de protecciones. Cada capa recibe
`forward(E, h, x)`:

- **E**: los embeddings de Qwen, sin posición.
- **h**: el estado oculto en ese punto.
- **x**: los números de pieza.

El injerto arranca neutro (W_o = 0): en el paso 0 no cambia ningún bit de lo que calcula Qwen. La escritura es
O(1) por pieza: la caché KV de Qwen se guarda en anillo y se acopla a los anillos de casilleros y a la memoria de
Hawkes de ASRN.

Los cálculos de VRAM y FLOPs, y por qué se eligió la opción A, están en [docs/CALCULOS.md](docs/CALCULOS.md).

## Instalar

```bash
pip install -r requirements.txt      # torch >= 2.9, transformers, asrn 1.0 (desde GitHub)
```

## Entrenar (A100 80 GB)

```bash
# piezas tokenizadas con el tokenizador de Qwen, int32 crudo (.u32/.i32) o .pt
python -m qwen_asrn.entrenar --datos train.u32 --validacion val.u32 \
    --largo 8192 --lote 1 --acumular 16 --pasos 2000 --lr 2e-4 --guardar_en qwen_asrn.pt
python -m qwen_asrn.entrenar ... --reanudar qwen_asrn.pt
```

Solo se guardan los pesos de ASRN (1.49 GB en fp32) y el estado de AdamW; Qwen se vuelve a cargar de su repositorio.

## Escribir

```bash
python -m qwen_asrn.escritura --asrn qwen_asrn.pt --texto "Había una vez" --n 200
```

```python
from qwen_asrn import QwenASRN, cargar_qwen, cargar_asrn, escribir
modelo = QwenASRN(cargar_qwen('Qwen/Qwen2.5-7B', 'cuda'))
cargar_asrn(modelo, 'qwen_asrn.pt')
y = escribir(modelo, x, n_nuevos=200)          # x: (B, L) piezas
```

## Pruebas

```bash
python -m pytest tests          # en CPU, con un Qwen2 diminuto de pesos al azar (unos 10 s)
python -m qwen_asrn.presupuesto --largo 8192
```

Las pruebas comprueban:

- La ida propia coincide con la de `transformers` y la atención por bloques coincide con una máscara densa.
- El arranque es neutro bit a bit en bf16 y el contrato de precisión se cumple.
- Solo ASRN recibe gradiente, y W_o aprende desde el paso 0.
- `checkpoint` da las mismas cuentas.
- La escritura O(1) da lo mismo que la ida entera, factorizada o no y con prefill por pasos o no.
- El presupuesto coincide con los parámetros reales y con los bytes que guarda autograd.

Todavía no se ha corrido con los pesos reales de Qwen 2.5 7B ni en una GPU: los números marcados "estimado" en el
documento hay que medirlos ahí.
