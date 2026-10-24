"""Qwen 2.5 congelado (bf16) + capas asrn.CapaASRNEstable (fp32) cada `largo_cada` bloques.

    from qwen_asrn import QwenASRN, cargar_qwen, escribir
    modelo = QwenASRN(cargar_qwen('Qwen/Qwen2.5-7B', 'cuda'))       # ventana 1024 + 4 sumideros, 14 capas ASRN
    perdida = modelo.perdida(x, y)                                   # entrenar: solo las capas ASRN tienen gradiente
    y = escribir(modelo, x, n_nuevos=200)                            # escritura O(1) por pieza

Entrenar: python -m qwen_asrn.entrenar --datos piezas.u32 ...   Presupuesto: python -m qwen_asrn.presupuesto
"""
__all__ = ['QwenASRN', 'EscritorQwenASRN', 'cargar_qwen', 'cargar_asrn', 'guardar_asrn', 'escribir',
           'verificar_arranque_neutro']


def __getattr__(nombre):          # importa torch solo al usarlo: presupuesto.py corre sin torch
    if nombre in ('EscritorQwenASRN', 'escribir'):
        from . import escritura
        return getattr(escritura, nombre)
    if nombre in __all__:
        from . import modelo
        return getattr(modelo, nombre)
    raise AttributeError(nombre)
