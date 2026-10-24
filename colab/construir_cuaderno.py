"""Genera colab/qwen_asrn_colab.ipynb: entrenar el injerto Qwen 2.5 7B + ASRN en Google Colab (A100).

    python colab/construir_cuaderno.py              # escribe el cuaderno
    python colab/construir_cuaderno.py --probar     # y lo ejecuta entero aqui en modo prueba (Qwen diminuto, CPU)

En modo prueba (QWEN_ASRN_PRUEBA=1) el cuaderno usa un Qwen2 diminuto de pesos al azar, texto sintetico y un
tokenizador de juguete, y toma el codigo de las carpetas QWEN_ASRN_DIR y ASRN_DIR en vez de GitHub: asi se revisa cada
celda sin GPU ni red.
"""
import json
import os
import subprocess
import sys
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(AQUI, 'qwen_asrn_colab.ipynb')
CELDAS = []


def md(texto):
    CELDAS.append({'cell_type': 'markdown', 'metadata': {}, 'source': texto.strip('\n').splitlines(True)})


def code(texto):
    CELDAS.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                   'source': texto.strip('\n').splitlines(True)})


md(r'''
# Qwen 2.5 7B + ASRN "1 y 1": entrenamiento en Colab

Qwen 2.5 7B queda **congelado en bf16**, con atención de ventana (1024 piezas + 4 sumideros). Después de los bloques
0, 2, …, 26 se injertan **14 capas `CapaASRNEstable` en fp32**, y solo esas capas entrenan, con AdamW. El arranque es
neutro: en el paso 0 el injerto no cambia ni un bit de lo que calcula Qwen, y el cuaderno lo comprueba antes de
entrenar. Los cálculos de memoria y velocidad están en `docs/CALCULOS.md` del repositorio `qwen-asrn`.

## Antes de empezar

1. **Entorno de ejecución → Cambiar tipo de entorno de ejecución → A100 GPU.** Hace falta Colab Pro o Pro+. Si la A100
   es de 40 GB, el cuaderno usa `checkpoint='todo'`, que entra con 8192 piezas pero va ~38 % más lento.
2. **Entorno de ejecución → Ejecutar todo.** El cuaderno pide permiso para montar Google Drive.

## Qué hay que subir

| qué | cuándo | cómo |
|---|---|---|
| **nada** | si los repositorios `KIROKOWA/ASRN` y `KIROKOWA/qwen-asrn` son públicos y usas `FUENTE = 'pg19'` o `'wikitext'` | el cuaderno descarga el código, Qwen (15 GB) y los textos |
| ZIP de `KIROKOWA/ASRN` (rama `main`) y ZIP de `KIROKOWA/qwen-asrn` (rama `claude/zealous-knuth-t4nhi3`) | si son **privados**: el cuaderno no puede clonarlos y te los pide | en GitHub: elige la rama → botón verde **Code → Download ZIP**; súbelos cuando aparezca el botón |
| tus textos `.txt` | con `FUENTE = 'texto'` | ponlos en `Mi unidad/qwen_asrn/datos/` (o súbelos cuando te los pida) |
| `train.u32` y `val.u32` | con `FUENTE = 'piezas'` (ya tokenizados) | en `Mi unidad/qwen_asrn/datos/`: piezas del tokenizador de Qwen 2.5, enteros de 32 bits crudos, `<\|endoftext\|>` (151643) entre documentos |

**Espacio en Drive:** unos 7 GB.

- `qwen_asrn.pt`, el punto de reanudación con los pesos de ASRN y AdamW: 4.5 GB.
- `qwen_asrn.mejor.pt`, solo los pesos: 1.5 GB.
- Los datos tokenizados: unos 100 MB por cada 25 M de piezas.

Si Colab se desconecta, **vuelve a ejecutar todo**: los datos ya tokenizados y el entrenamiento siguen desde Drive,
en el último paso guardado.

## Tiempo

Con los ajustes por defecto son 300 pasos × 8 × 8192 = 19.7 M de piezas. Eso da unas 1.5 h en una A100 de 80 GB o
unas 2 h en una de 40 GB, más unos 10 min de descargas y tokenización. Son estimaciones (aprovechamiento de la GPU
del 40 %): la celda de presupuesto calcula la tuya, y el registro muestra la velocidad real.
''')

code(r'''
# ===== Ajustes: cambia aqui =====
MODELO = 'Qwen/Qwen2.5-7B'
REPO_ASRN, RAMA_ASRN = 'https://github.com/KIROKOWA/ASRN.git', 'main'
REPO_QWEN_ASRN, RAMA_QWEN_ASRN = 'https://github.com/KIROKOWA/qwen-asrn.git', 'claude/zealous-knuth-t4nhi3'
CARPETA_DRIVE = 'qwen_asrn'        # en Mi unidad: datos, pesos y registro (sobreviven a una desconexion)

FUENTE = 'pg19'                    # 'pg19' libros largos (lo mejor para ASRN: citas lejanas) | 'wikitext' |
                                   # 'texto' tus .txt | 'piezas' tus train.u32 / val.u32 ya tokenizados
MAX_PIEZAS = 25_000_000            # piezas de entrenamiento a tokenizar (pg19 / wikitext / texto)
PIEZAS_VALIDACION = 1_000_000

PASOS = 300                        # pasos de AdamW
ACUMULAR = 8                       # ventanas por paso (lote efectivo = ACUMULAR x LARGO piezas)
LR = 2e-4
VENTANA, SUMIDEROS = 1024, 4       # atencion de Qwen: ultimas VENTANA piezas + SUMIDEROS primeras
LARGO = None                       # None: lo elige el cuaderno segun la memoria de la GPU
CHECKPOINT = None                  # None: 'asrn' con 80 GB, 'todo' con 40 GB
GUARDAR_CADA, EVALUAR_CADA, LOTES_EVAL = 50, 50, 8
''')

code(r'''
# Entorno: GPU, Drive y carpetas
import os, sys, json, glob, shutil, subprocess, time, zipfile
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # menos fragmentacion
PRUEBA = os.environ.get('QWEN_ASRN_PRUEBA') == '1'           # modo prueba: Qwen diminuto en CPU (ver el generador)
EN_COLAB = 'google.colab' in sys.modules
if shutil.which('nvidia-smi'):
    print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip())
elif not PRUEBA:
    raise RuntimeError('No hay GPU. En Colab: Entorno de ejecucion -> Cambiar tipo de entorno -> A100 GPU')
CARPETA = os.path.abspath(os.environ.get('QWEN_ASRN_SALIDA', 'qwen_asrn_salida'))
if EN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    CARPETA = os.path.join('/content/drive/MyDrive', CARPETA_DRIVE)
DATOS = os.path.join(CARPETA, 'datos')
os.makedirs(DATOS, exist_ok=True)
ARCH_PESOS = os.path.join(CARPETA, 'qwen_asrn.pt')
ARCH_REGISTRO = os.path.join(CARPETA, 'registro.txt')
print('Todo se guarda en', CARPETA)
''')

code(r'''
# Librerias: PyTorch >= 2.9 (lo pide asrn), transformers, datasets
import importlib.metadata as _md

def _pip(*paquetes):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *paquetes], check=True)

def _version(paquete):
    try:
        return tuple(int(x) for x in _md.version(paquete).split('+')[0].split('.')[:2])
    except Exception:                                           # noqa: BLE001
        return (0, 0)

if _version('torch') < (2, 9):
    # torchvision / torchaudio de Colab quedarian desparejados con el PyTorch nuevo (y transformers los importa)
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-q', '-y', 'torchvision', 'torchaudio'])
    _pip('torch>=2.9')
if not PRUEBA:
    _pip('transformers>=4.45', 'safetensors', 'datasets', 'pytest')
import torch, transformers
print('torch', torch.__version__, '| transformers', transformers.__version__,
      '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no')
''')

code(r'''
# El codigo: asrn (KIROKOWA/ASRN) y qwen_asrn (KIROKOWA/qwen-asrn). De GitHub si son publicos; si no, de un ZIP.
BASE = '/content' if EN_COLAB else os.path.abspath('codigo')
DIR_ASRN, DIR_QA = os.path.join(BASE, 'ASRN'), os.path.join(BASE, 'qwen-asrn')

def _traer(repo, rama, destino, marca, variable):
    if os.path.exists(os.path.join(destino, marca)):
        return
    shutil.rmtree(destino, ignore_errors=True)
    if os.environ.get(variable):                                # modo prueba: una carpeta local
        shutil.copytree(os.environ[variable], destino, ignore=shutil.ignore_patterns('.git', '__pycache__'))
        return
    r = subprocess.run(['git', 'clone', '-q', '--depth', '1', '-b', rama, repo, destino], capture_output=True,
                       text=True, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
    if r.returncode == 0:
        return
    if not EN_COLAB:
        raise RuntimeError(f'No se pudo clonar {repo} ({rama}): {r.stderr.strip()}')
    print(f'No se pudo clonar {repo} rama {rama} (¿privado?). Sube su ZIP (GitHub: rama {rama} -> Code -> '
          'Download ZIP):')
    from google.colab import files
    subido = files.upload()
    if not subido:
        raise RuntimeError('No se subio ningun archivo')
    tmp = destino + '_zip'
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(next(iter(subido))) as z:
        z.extractall(tmp)
    raices = sorted(glob.glob(os.path.join(tmp, '**', marca), recursive=True), key=len)
    if not raices:
        raise RuntimeError(f'Ese ZIP no tiene {marca}: hace falta el ZIP de {repo}, rama {rama}')
    raiz = raices[0][:-len(marca)]
    shutil.copytree(raiz, destino)

_traer(REPO_ASRN, RAMA_ASRN, DIR_ASRN, 'asrn/estable.py', 'ASRN_DIR')
_traer(REPO_QWEN_ASRN, RAMA_QWEN_ASRN, DIR_QA, 'qwen_asrn/modelo.py', 'QWEN_ASRN_DIR')
sys.path[:0] = [DIR_QA, DIR_ASRN, os.path.join(DIR_QA, 'tests')]
import asrn, qwen_asrn
print('asrn', asrn.__version__, 'en', DIR_ASRN, '| qwen_asrn en', DIR_QA)
''')

code(r'''
# Pruebas rapidas (CPU, Qwen2 diminuto, ~10 s): si algo del codigo esta mal, se ve aqui y no a la hora de entrenar
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests'], cwd=DIR_QA,
                   capture_output=True, text=True,
                   env={**os.environ, 'PYTHONPATH': os.pathsep.join([DIR_QA, DIR_ASRN]), 'CUDA_VISIBLE_DEVICES': ''})
print(r.stdout[-1500:])
if r.returncode != 0:
    print(r.stderr[-3000:])
    raise RuntimeError('fallan las pruebas de qwen_asrn')
''')

code(r'''
# Datos: piezas del tokenizador de Qwen en DATOS/train.u32 y DATOS/val.u32 (se hace una vez; queda en Drive)
import numpy as np
from transformers import AutoTokenizer

class _TokPrueba:                                               # modo prueba: un tokenizador de juguete
    eos_token_id = 0
    def __call__(self, textos, add_special_tokens=False):
        return {'input_ids': [[1 + ord(c) % 299 for c in t] for t in textos]}
    def encode(self, t):
        return self([t])['input_ids'][0]
    def decode(self, ids):
        return ''.join(chr(96 + i % 27) for i in ids)

tok = _TokPrueba() if PRUEBA else AutoTokenizer.from_pretrained(MODELO)
EOS = tok.eos_token_id
ARCH_T, ARCH_V = os.path.join(DATOS, 'train.u32'), os.path.join(DATOS, 'val.u32')

def _textos():
    if PRUEBA:
        import random
        rnd = random.Random(0)
        palabras = ['casa', 'rio', 'luz', 'mar', 'sol', 'pan', 'voz', 'tren', 'lago', 'nube']
        while True:
            parrafo = ' '.join(rnd.choice(palabras) for _ in range(60))
            yield parrafo + ' ' + ' '.join(rnd.choice(palabras) for _ in range(200)) + ' ' + parrafo
    from datasets import load_dataset
    fuente = FUENTE
    if fuente == 'pg19':
        try:
            libros = load_dataset('emozilla/pg19', split='train', streaming=True)
        except Exception as e:                                  # noqa: BLE001
            print('No se pudo abrir pg19 (', e, '): uso wikitext-103')
            fuente = 'wikitext'
        else:
            for fila in libros:
                yield fila['text']
    if fuente == 'wikitext':                                   # filas = lineas: se juntan en trozos de ~100 KB
        trozo = []
        for fila in load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train', streaming=True):
            trozo.append(fila['text'])
            if sum(map(len, trozo)) > 100_000:
                yield ''.join(trozo)
                trozo = []
    elif fuente == 'texto':
        archivos = sorted(glob.glob(os.path.join(DATOS, '*.txt')))
        if not archivos and EN_COLAB:
            from google.colab import files
            print('Sube tus .txt:')
            for nombre, contenido in files.upload().items():
                with open(os.path.join(DATOS, os.path.basename(nombre)), 'wb') as f:
                    f.write(contenido)
            archivos = sorted(glob.glob(os.path.join(DATOS, '*.txt')))
        if not archivos:
            raise RuntimeError(f'No hay .txt en {DATOS}')
        for a in archivos:
            with open(a, encoding='utf-8', errors='replace') as f:
                yield f.read()
    elif fuente != 'pg19':
        raise ValueError(f'FUENTE desconocida: {FUENTE!r}')

def _tokenizar(objetivo):
    partes, total, lote, t0 = [], 0, [], time.time()
    def vaciar():
        nonlocal total
        for ids in tok(lote, add_special_tokens=False)['input_ids']:
            partes.append(np.asarray(ids + [EOS], dtype=np.uint32))
            total += len(ids) + 1
        lote.clear()
    for texto in _textos():
        lote.append(texto)
        if len(lote) == 16:
            vaciar()
            print(f'\r{total:,} piezas ({time.time() - t0:.0f} s)', end='')
            if total >= objetivo:
                break
    if lote:
        vaciar()
    print()
    return np.concatenate(partes)

if FUENTE == 'piezas':
    if not (os.path.exists(ARCH_T) and os.path.exists(ARCH_V)):
        raise RuntimeError(f'Con FUENTE="piezas" sube train.u32 y val.u32 a {DATOS}')
elif not (os.path.exists(ARCH_T) and os.path.exists(ARCH_V)):
    if PRUEBA:
        MAX_PIEZAS, PIEZAS_VALIDACION = 20_000, 4_000
    todo = _tokenizar(MAX_PIEZAS + PIEZAS_VALIDACION)
    todo[:-PIEZAS_VALIDACION].tofile(ARCH_T + '.tmp')
    todo[-PIEZAS_VALIDACION:].tofile(ARCH_V + '.tmp')
    os.replace(ARCH_T + '.tmp', ARCH_T)
    os.replace(ARCH_V + '.tmp', ARCH_V)
datos, validacion = asrn.leer_piezas(ARCH_T), asrn.leer_piezas(ARCH_V)
print(f'entrenamiento {datos.numel():,} piezas | validacion {validacion.numel():,} | maximo {int(datos.max())}')
''')

code(r'''
# Presupuesto para esta GPU: elige LARGO y CHECKPOINT si quedaron en None (qwen_asrn/presupuesto.py)
from qwen_asrn import presupuesto as P
if PRUEBA:
    VENTANA, SUMIDEROS, LARGO, CHECKPOINT, ACUMULAR, GUARDAR_CADA, EVALUAR_CADA, LOTES_EVAL = 16, 2, 64, 'asrn', 2, 3, 3, 2
    PASOS = int(os.environ.get('QWEN_ASRN_PASOS', 6))
mem = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 80 * P.GIB
q, inj = P.ConfigQwen(), P.ConfigInjerto(ventana=VENTANA, sumideros=SUMIDEROS)
if LARGO is None or CHECKPOINT is None:
    for ck, L in (('asrn', 8192), ('todo', 8192), ('asrn', 4096), ('todo', 4096), ('todo', 2048)):
        if (CHECKPOINT in (None, ck) and LARGO in (None, L)
                and 1.15 * L <= P.piezas_maximas(q, inj, 0.97 * mem, ck)):
            LARGO, CHECKPOINT = L, ck
            break
    else:
        raise RuntimeError(f'No entra ni con 2048 piezas en {mem / P.GIB:.0f} GiB')
assert LARGO > VENTANA, 'LARGO tiene que pasar de VENTANA: si no, ASRN no ve nada que la atencion no vea'
m = P.memoria_entrenamiento(q, inj, LARGO, CHECKPOINT)
f = P.flops_por_pieza(q, inj, CHECKPOINT)
piezas_s = 312e12 * 0.40 / f['entrenamiento']
horas = PASOS * ACUMULAR * LARGO / piezas_s / 3600
print(f'GPU de {mem / P.GIB:.0f} GiB -> LARGO={LARGO}, CHECKPOINT={CHECKPOINT!r}')
for k, v in m.items():
    print(f'  {k:42s} {v / P.GIB:6.2f} GiB')
print(f'{f["entrenamiento"] / 1e9:.1f} GFLOP por pieza -> ~{piezas_s:,.0f} piezas/s (estimado, MFU 40 %): '
      f'{PASOS} pasos x {ACUMULAR} x {LARGO} = {PASOS * ACUMULAR * LARGO / 1e6:.1f} M piezas, ~{horas:.1f} h '
      f'(~{13 * horas:.0f} unidades de computo de Colab)')
''')

code(r'''
# El modelo: Qwen 2.5 7B en bf16 congelado + 14 CapaASRNEstable en fp32 (W_o = 0: arranque neutro)
from qwen_asrn import QwenASRN, cargar_qwen
if PRUEBA:
    from test_qwen_asrn import OPC, qwen_chico
    qwen, opciones = qwen_chico(torch.bfloat16), dict(opciones_capa=OPC)
else:
    qwen, opciones = cargar_qwen(MODELO, 'cuda'), {}
modelo = QwenASRN(qwen, ventana=VENTANA, sumideros=SUMIDEROS, checkpoint=CHECKPOINT, **opciones)
n = sum(p.numel() for p in modelo.parametros_entrenables())
print(f'{len(modelo.pos_largo)} capas ASRN despues de los bloques {modelo.pos_largo}: {n:,} parametros fp32 '
      f'entrenables | Qwen congelado: {sum(p.numel() for p in modelo.qwen.parameters()):,}')
if torch.cuda.is_available():
    print(f'memoria en uso: {torch.cuda.memory_allocated() / 2 ** 30:.1f} GiB')
''')

code(r'''
# Linea de base: perdida de validacion de Qwen solo (con la misma ventana). Se calcula una vez y queda en Drive.
from qwen_asrn.entrenar import evaluar
ARCH_BASE = os.path.join(CARPETA, 'base.json')
if os.path.exists(ARCH_BASE):
    BASE_VAL = json.load(open(ARCH_BASE))['qwen_solo']
else:
    BASE_VAL = evaluar(modelo, validacion, LARGO, 1, LOTES_EVAL, sin_asrn=True)
    json.dump({'qwen_solo': BASE_VAL, 'largo': LARGO, 'ventana': VENTANA}, open(ARCH_BASE, 'w'))
print(f'Qwen solo (ventana {VENTANA}): perdida de validacion {BASE_VAL:.4f}')
''')

code(r'''
# Entrenar. Reanuda solo si ya hay qwen_asrn.pt en Drive; si no, comprueba antes el arranque neutro (bit a bit).
from qwen_asrn.entrenar import entrenar
_log = open(ARCH_REGISTRO, 'a')

def registro(texto):
    print(texto)
    _log.write(time.strftime('%Y-%m-%d %H:%M:%S ') + texto + '\n')
    _log.flush()

registro(f'--- LARGO={LARGO} CHECKPOINT={CHECKPOINT} PASOS={PASOS} ACUMULAR={ACUMULAR} LR={LR} '
         f'VENTANA={VENTANA} SUMIDEROS={SUMIDEROS} FUENTE={FUENTE}')
entrenar(modelo, datos, validacion, largo=LARGO, lote=1, acumular=ACUMULAR, pasos=PASOS, lr=LR,
         calentamiento=max(1, PASOS // 20), evaluar_cada=EVALUAR_CADA, lotes_eval=LOTES_EVAL, guardar_en=ARCH_PESOS,
         guardar_cada=GUARDAR_CADA, reanudar=ARCH_PESOS if os.path.exists(ARCH_PESOS) else None,
         registrar_cada=1 if PRUEBA else 5, registro=registro)
''')

code(r'''
# Resultado: validacion con ASRN contra Qwen solo, y una prueba de cita lejana (mas alla de la ventana)
import torch.nn.functional as F
from qwen_asrn import cargar_asrn
ARCH_MEJOR = os.path.splitext(ARCH_PESOS)[0] + '.mejor.pt'
if os.path.exists(ARCH_MEJOR):
    cargar_asrn(modelo, ARCH_MEJOR)
con = evaluar(modelo, validacion, LARGO, 1, LOTES_EVAL)
registro(f'validacion: Qwen solo {BASE_VAL:.4f} | Qwen + ASRN {con:.4f} | diferencia {con - BASE_VAL:+.4f}')

@torch.no_grad()
def cita_lejana():
    """[A][relleno][A] en LARGO piezas: con el relleno mas largo que la ventana, la atencion ya no ve la primera A.
    Perdida media sobre la segunda A (despues de una pista de n_pista piezas) con y sin ASRN."""
    n_cita = min(128, LARGO // 4)
    n_pista = max(1, n_cita // 8)
    modelo.eval()
    v = validacion.long()
    A = v[:n_cita]
    relleno = v[n_cita:n_cita + LARGO - 2 * n_cita]
    x = torch.cat([A, relleno, A]).unsqueeze(0).to(modelo.qwen.model.embed_tokens.weight.device)
    ini = n_cita + relleno.numel() + n_pista
    r = {}
    for nombre, sin in (('Qwen solo', True), ('Qwen + ASRN', False)):
        h = modelo.ocultos(x[:, :-1], sin_asrn=sin)                     # logits solo de la segunda A
        r[nombre] = float(F.cross_entropy(modelo.logits(h[0, ini - 1:]).float(), x[0, ini:]))
    return r, relleno.numel()

r, dist = cita_lejana()
registro(f'cita a {dist:,} piezas (ventana {VENTANA}): ' + ' | '.join(f'{k} {v:.3f}' for k, v in r.items()))
''')

code(r'''
# Escribir (escritura O(1) por pieza: cache KV en anillo + anillos de casilleros + Hawkes)
from qwen_asrn import escribir
inicio = 'Había una vez' if not PRUEBA else 'casa rio luz'
x = torch.tensor([tok.encode(inicio)])
y = escribir(modelo, x, n_nuevos=80 if not PRUEBA else 10, parar_en=EOS)
print(inicio + tok.decode(y[0, x.shape[1]:].tolist()))
''')

md(r'''
## Archivos

Quedan en `Mi unidad/qwen_asrn/`:

- `qwen_asrn.mejor.pt`: los pesos de ASRN con la mejor validación, 1.5 GB. Es el archivo para escribir.
- `qwen_asrn.pt`: para reanudar el entrenamiento. Incluye AdamW, 4.5 GB.
- `registro.txt`: todo lo que se imprimió.
- `base.json`: la pérdida de validación de Qwen solo.

Qwen no se guarda, porque se vuelve a cargar de su repositorio. Para escribir fuera de Colab:
`python -m qwen_asrn.escritura --asrn qwen_asrn.mejor.pt --texto "..."`.
''')


def construir():
    nb = {'cells': CELDAS, 'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'A100', 'machine_shape': 'hm', 'provenance': []},
                       'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                       'language_info': {'name': 'python'}}}
    with open(SALIDA, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write('\n')
    print('escrito', SALIDA)


def probar():
    """Ejecuta todas las celdas en orden, en un solo espacio de nombres, en modo prueba."""
    raiz = os.path.dirname(AQUI)
    asrn_dir = os.environ.get('ASRN_DIR') or os.path.join(os.path.dirname(raiz), 'ASRN')
    tmp = tempfile.mkdtemp()
    os.environ.update(QWEN_ASRN_PRUEBA='1', QWEN_ASRN_DIR=raiz, ASRN_DIR=asrn_dir,
                      QWEN_ASRN_SALIDA=os.path.join(tmp, 'salida'))
    os.chdir(tmp)
    ns = {'__name__': '__main__'}
    for k, c in enumerate(c for c in CELDAS if c['cell_type'] == 'code'):
        print(f'\n===== celda {k + 1} =====')
        exec(compile(''.join(c['source']), f'celda_{k + 1}', 'exec'), ns)
    print('\n===== segunda vuelta (como tras una desconexion: datos y pesos ya en la carpeta) =====')
    os.environ['QWEN_ASRN_PASOS'] = '8'
    for k, c in enumerate(c for c in CELDAS if c['cell_type'] == 'code'):
        exec(compile(''.join(c['source']), f'celda_{k + 1}', 'exec'), ns)


if __name__ == '__main__':
    construir()
    if '--probar' in sys.argv:
        probar()
