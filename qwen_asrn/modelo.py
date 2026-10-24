"""Qwen 2.5 congelado + CapaASRNEstable cada `largo_cada` bloques (arquitectura "1 y 1" con largo_cada=2).

    h_0 = E = embed_tokens(x)                                  (B, L, d) bf16, sin posicion (Qwen usa RoPE)
    para cada bloque i de Qwen:
        h = bloque_i(h)                                        atencion con ventana W + S sumideros, SwiGLU
        si i en pos_largo:  h = CapaASRNEstable_j(E, h, x)     en fp32, con bf16 solo por autocast
    logits = lm_head(norm(h))

Como asrn.ModeloASRN: una capa ASRN despues del bloque i si i % largo_cada == 0, sin contar el ultimo; en Qwen 2.5 7B
(28 bloques) son 14 capas, despues de los bloques 0, 2, ..., 26.

Contrato de precision (revisar_contrato(), en cada ida):
  - Qwen: pesos en bf16 y congelados (requires_grad=False): embeddings, RMSNorms, atencion y SwiGLU de los 28 bloques.
    El estado h entre bloques va en bf16, como en Qwen.
  - ASRN: pesos estrictamente en fp32 (y los buffers c_vistos en fp64 y r_firma en fp32). La capa recibe E y h
    convertidos a fp32 (exacto desde bf16) y corre dentro de torch.autocast(bf16): ahi bajan a bf16 solo las
    multiplicaciones que autocast baja; el reloj sigue en fp64 (reloj_64), su proyeccion en fp32 de verdad
    (_ProyeccionIEEE, sin TF32 aunque este encendido) y la puerta en fp32. La salida vuelve a bf16 para el bloque
    siguiente.

Arranque neutro exacto: h' = h + g * W_o c con W_o = 0 da h' = h bit a bit (h fp32 + 0 = h, y bf16 -> fp32 -> bf16
es exacto), sea cual sea la puerta g. Por eso se anula W_o y no la puerta: sigmoid nunca da 0 exacto, y una puerta
cerrada a la fuerza (sesgo muy negativo) mata el gradiente. Con W_o = 0, dL/dW_o = sum_t g_t (dL/dh'_t) c_t^T no es
cero en el paso 0 (como B = 0 en LoRA). La puerta g arranca tambien en 0 (g = 1/2): los estados de Qwen tienen
canales enormes y el g aleatorio de nn.Linear saturaria la sigmoide. verificar_arranque_neutro() lo comprueba.

Atencion con ventana: cada pieza ve las ultimas W piezas (ella incluida) y los primeros S "sumideros" (StreamingLLM:
sin ellos Qwen se degrada en cuanto el texto pasa de la ventana). Con L <= W es exactamente la atencion causal de
Qwen; con L > W el modelo de partida es Qwen-con-ventana, no Qwen (Qwen 2.5 se entreno con atencion completa). Se
calcula por bloques de W preguntas contra 2W + 8 llaves con scaled_dot_product_attention: O(L W), sin repetir las
cabezas de k y v (las G = 28/4 cabezas de pregunta de cada cabeza kv van apiladas en el largo).

checkpoint: 'asrn' (recalcula las capas ASRN en la vuelta), 'todo' (tambien los bloques de Qwen) o 'no'. Los bloques
de Qwen congelados guardan para la vuelta solo lo que hace falta para dX (presupuesto.py lo cuenta byte a byte).
"""
import contextlib
import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from asrn import CapaASRNEstable, Protecciones
from asrn import nucleo as v89

RANURAS_SUMIDERO = 8          # ranuras de sumidero en las llaves (las que sobran van enmascaradas): multiplo de 8
_CHECKPOINTS = ('no', 'asrn', 'todo')


# ---------------------------------------------------------------------------------------------------------------
# Piezas de Qwen con menos memoria para la vuelta (mismas cuentas en la ida)
# ---------------------------------------------------------------------------------------------------------------

def _a32(x):
    """x en fp32 si es de 16 bits; fp32 y fp64 quedan como estan (en bf16, lo mismo que x.to(torch.float32))."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


class _RMSNorm(torch.autograd.Function):
    """w * (x / rms(x)) con las mismas operaciones que Qwen2RMSNorm (x a fp32, varianza, rsqrt, de vuelta al tipo de
    x, por w). Para la vuelta guarda x en su tipo (bf16: 2 bytes por numero) en vez de la copia en fp32."""

    @staticmethod
    def forward(ctx, x, w, eps):
        x32 = _a32(x)
        r = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
        ctx.save_for_backward(x, w)
        ctx.eps = eps
        return w * (x32 * r).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        x32 = _a32(x)
        r = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + ctx.eps)
        n = x32 * r
        gx = gw = None
        if ctx.needs_input_grad[0]:
            gn = _a32(g * w)
            gx = (r * (gn - n * (gn * n).mean(-1, keepdim=True))).to(x.dtype)
        if ctx.needs_input_grad[1]:
            gw = (g * n.to(x.dtype)).reshape(-1, g.shape[-1]).sum(0).to(w.dtype)
        return gx, gw, None


def _rmsnorm(x, modulo):
    return _RMSNorm.apply(x, modulo.weight, modulo.variance_epsilon)


class _SwiGLU(torch.autograd.Function):
    """silu(a) * b. Para la vuelta guarda a y b (no tambien silu(a)): 4 bytes por numero del MLP en vez de 6."""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return F.silu(a) * b

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        a32, g32 = _a32(a), _a32(g)
        s = torch.sigmoid(a32)
        ga = (g32 * _a32(b) * s * (1 + a32 * (1 - s))).to(a.dtype) if ctx.needs_input_grad[0] else None
        gb = (g32 * a32 * s).to(b.dtype) if ctx.needs_input_grad[1] else None
        return ga, gb


def rope(posiciones, dim_cabeza, theta, dtype):
    """cos, sin (L, dim_cabeza) de RoPE para posiciones absolutas (L,). Los angulos se calculan en fp64: con 1 M de
    piezas, pos * inv_freq en fp32 pierde ~0.06 rad en la frecuencia mas alta (Qwen los calcula en fp32; con
    posiciones chicas, al pasar a bf16, da los mismos numeros salvo rarisimos empates de redondeo)."""
    inv = 1.0 / (theta ** (torch.arange(0, dim_cabeza, 2, dtype=torch.float64, device=posiciones.device) / dim_cabeza))
    ang = posiciones.to(torch.float64).unsqueeze(-1) * inv
    ang = torch.cat([ang, ang], -1)
    return ang.cos().to(dtype), ang.sin().to(dtype)


def _rotar(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def mascara_bloques(L, ventana, sumideros, G, dispositivo):
    """(1, nb, G*Wb, 8 + 2Wb) bool para _atencion_bloques. Bloque i, pregunta a (posicion p = i*Wb + a); llaves:
      sumidero s:          visible si s < S y s <= p - W (fuera de la ventana: dentro ya la ve como pieza normal)
      bloque anterior j:   posicion (i-1)W + j, visible si p - W < posicion, o sea j > a (y i >= 1)
      bloque actual j:     visible si j <= a (causal)
    Wb = W si L > W; si no, un solo bloque de L (atencion causal comun). Las G cabezas de pregunta que comparten una
    cabeza kv van apiladas en las filas."""
    Wb = ventana if L > ventana else L
    nb = -(-L // Wb)
    i = torch.arange(nb, device=dispositivo).view(nb, 1, 1)
    a = torch.arange(Wb, device=dispositivo).view(1, Wb, 1)
    j = torch.arange(Wb, device=dispositivo).view(1, 1, Wb)
    s = torch.arange(RANURAS_SUMIDERO, device=dispositivo).view(1, 1, -1)
    p = i * Wb + a
    sumidero = (s < sumideros) & (s <= p - ventana)
    anterior = (j > a) & (i >= 1)
    actual = (j <= a).expand(nb, Wb, Wb)
    m = torch.cat([sumidero, anterior.expand(nb, Wb, Wb), actual], -1)            # (nb, Wb, 8 + 2Wb)
    return m.repeat(1, G, 1).unsqueeze(0)


def _atencion_bloques(q, k, v, ventana, sumideros, mascara):
    """q (B, Hq, L, hd), k y v (B, Hk, L, hd) ya con RoPE -> (B, Hq, L, hd). Bloques de Wb preguntas contra
    [8 sumideros | bloque anterior | bloque actual]; mascara de mascara_bloques()."""
    B, Hq, L, hd = q.shape
    Hk = k.shape[1]
    G = Hq // Hk
    Wb = ventana if L > ventana else L
    nb = -(-L // Wb)
    relleno = nb * Wb - L
    if relleno:                    # preguntas de relleno (se descartan) y llaves de relleno (siempre en el futuro)
        q, k, v = (F.pad(t, (0, 0, 0, relleno)) for t in (q, k, v))
    qb = q.view(B, Hk, G, nb, Wb, hd).permute(0, 1, 3, 2, 4, 5).reshape(B * Hk, nb, G * Wb, hd)
    n_s = min(sumideros, L)

    def llaves(t):
        tb = t.reshape(B * Hk, nb, Wb, hd)
        anterior = F.pad(tb[:, :-1], (0, 0, 0, 0, 1, 0))                          # bloque i-1 (ceros en el 0)
        sum_ = F.pad(t[:, :, :n_s], (0, 0, 0, RANURAS_SUMIDERO - n_s)).reshape(B * Hk, 1, RANURAS_SUMIDERO, hd)
        return torch.cat([sum_.expand(-1, nb, -1, -1), anterior, tb], 2)            # (B*Hk, nb, 8 + 2Wb, hd)

    o = F.scaled_dot_product_attention(qb, llaves(k), llaves(v), attn_mask=mascara)  # (B*Hk, nb, G*Wb, hd)
    o = o.view(B, Hk, nb, G, Wb, hd).permute(0, 1, 3, 2, 4, 5).reshape(B, Hq, nb * Wb, hd)
    return o[:, :, :L]


# ---------------------------------------------------------------------------------------------------------------
# El modelo
# ---------------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def _tipo_por_omision(tipo):
    antes = torch.get_default_dtype()
    torch.set_default_dtype(tipo)
    try:
        yield
    finally:
        torch.set_default_dtype(antes)


def _theta_rope(cfg):
    theta = getattr(cfg, 'rope_theta', None)
    parametros = getattr(cfg, 'rope_parameters', None) or {}
    escala = getattr(cfg, 'rope_scaling', None) or {}
    tipo = parametros.get('rope_type', escala.get('rope_type', escala.get('type', 'default')))
    if tipo not in (None, 'default'):
        raise NotImplementedError(f"RoPE de tipo {tipo!r}: solo esta el RoPE de Qwen 2.5 sin escalado")
    return float(theta if theta is not None else parametros['rope_theta'])


class QwenASRN(nn.Module):
    """qwen: un Qwen2ForCausalLM de transformers (cargar_qwen()). Sus pesos se congelan; solo `largos` (las capas
    CapaASRNEstable, fp32) entrenan. contrato='estricto' exige Qwen en bf16 (contrato='libre': cualquier tipo, para
    pruebas en CPU con fp32)."""

    def __init__(self, qwen, ventana=1024, sumideros=4, largo_cada=2, protecciones=None, checkpoint='asrn',
                 trozo_perdida=1024, precision_asrn='bf16', contrato='estricto', opciones_capa=None):
        super().__init__()
        cfg = qwen.config
        if getattr(cfg, 'hidden_act', 'silu') != 'silu':
            raise NotImplementedError(f"hidden_act={cfg.hidden_act!r}: el MLP de Qwen 2.5 es SwiGLU (silu)")
        assert 1 <= ventana and 0 <= sumideros <= RANURAS_SUMIDERO and largo_cada >= 1
        assert checkpoint in _CHECKPOINTS and precision_asrn in ('bf16', 'fp32') and contrato in ('estricto', 'libre')
        self.qwen = qwen
        self.d = cfg.hidden_size
        self.V = cfg.vocab_size
        assert self.V <= 2 ** 24, "las firmas por numero de pieza de ASRN son exactas hasta 2**24 piezas"
        self.n_capas = cfg.num_hidden_layers
        self.cabezas, self.cabezas_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.dim_cabeza = getattr(cfg, 'head_dim', None) or self.d // self.cabezas
        self.theta = _theta_rope(cfg)
        self.ventana, self.sumideros, self.largo_cada = ventana, sumideros, largo_cada
        self.checkpoint, self.trozo_perdida, self.precision_asrn, self.contrato = (checkpoint, trozo_perdida,
                                                                                  precision_asrn, contrato)
        self.pos_largo = [i for i in range(self.n_capas - 1) if i % largo_cada == 0]
        self.protecciones = protecciones or Protecciones()
        self.opciones_capa = dict(opciones_capa or {})
        for p in qwen.parameters():
            p.requires_grad_(False)
        disp = qwen.model.embed_tokens.weight.device
        with _tipo_por_omision(torch.float32):
            self.largos = nn.ModuleList([CapaASRNEstable(self.d, self.protecciones, **self.opciones_capa)
                                         for _ in self.pos_largo]).to(disp)
        with torch.no_grad():
            for capa in self.largos:              # arranque neutro exacto
                capa.o.weight.zero_()
                capa.g.weight.zero_()
                capa.g.bias.zero_()
        self.revisar_contrato(qwen_tambien=True)

    # --- config y contrato --------------------------------------------------------------------------------------

    @property
    def config(self):
        return {'formato': 'qwen-asrn-1', 'qwen': getattr(self.qwen.config, '_name_or_path', ''), 'd': self.d,
                'V': self.V, 'n_capas': self.n_capas, 'ventana': self.ventana, 'sumideros': self.sumideros,
                'largo_cada': self.largo_cada, 'pos_largo': list(self.pos_largo),
                'protecciones': dataclasses.asdict(self.protecciones), 'opciones_capa': dict(self.opciones_capa)}

    def revisar_contrato(self, qwen_tambien=False):
        """ASRN en fp32 (c_vistos en fp64); con qwen_tambien, Qwen congelado y (contrato estricto) en bf16."""
        for nombre, p in self.largos.named_parameters():
            if p.dtype != torch.float32:
                raise ValueError(f"largos.{nombre} esta en {p.dtype}: los pesos de ASRN van en fp32 (bf16 solo por "
                                 "autocast; no conviertas el modelo con .to(torch.bfloat16))")
        for capa in self.largos:
            if capa.c_vistos.dtype != torch.float64 or capa.r_firma.dtype != torch.float32:
                raise ValueError("los buffers de ASRN cambiaron de tipo (c_vistos va en fp64, r_firma en fp32)")
        if qwen_tambien:
            for nombre, p in self.qwen.named_parameters():
                if p.requires_grad:
                    raise ValueError(f"qwen.{nombre} no esta congelado")
                if self.contrato == 'estricto' and p.dtype != torch.bfloat16:
                    raise ValueError(f"qwen.{nombre} esta en {p.dtype}: el contrato estricto pide Qwen en bf16")

    def parametros_entrenables(self):
        return [p for p in self.largos.parameters() if p.requires_grad]

    # --- piezas ------------------------------------------------------------------------------------------------

    def _bloque(self, i, h, cos, sin, mascara, kv=False):
        """Bloque i de Qwen sobre h (B, L, d) -> h (y k, v con RoPE (B, Hk, L, hd) si kv)."""
        c = self.qwen.model.layers[i]
        a = c.self_attn
        B, L, _ = h.shape
        x = _rmsnorm(h, c.input_layernorm)
        q = a.q_proj(x).view(B, L, self.cabezas, self.dim_cabeza).transpose(1, 2)
        k = a.k_proj(x).view(B, L, self.cabezas_kv, self.dim_cabeza).transpose(1, 2)
        v = a.v_proj(x).view(B, L, self.cabezas_kv, self.dim_cabeza).transpose(1, 2)
        q, k = _rotar(q, cos, sin), _rotar(k, cos, sin)
        o = _atencion_bloques(q, k, v, self.ventana, self.sumideros, mascara)
        h = h + a.o_proj(o.transpose(1, 2).reshape(B, L, -1))
        m = c.mlp
        x = _rmsnorm(h, c.post_attention_layernorm)
        h = h + m.down_proj(_SwiGLU.apply(m.gate_proj(x), m.up_proj(x)))
        return (h, k, v) if kv else h

    def autocast_asrn(self, dispositivo, precision=None):
        precision = precision or self.precision_asrn
        return torch.autocast(device_type=dispositivo.type, dtype=torch.bfloat16, enabled=precision == 'bf16')

    def _asrn(self, j, E, h, x):
        """CapaASRNEstable j: forward(E, h, x) en fp32 (+ autocast bf16), vuelta al tipo de h."""
        with self.autocast_asrn(h.device):
            h2 = self.largos[j](E.float(), h.float(), x)
        return h2.to(h.dtype)

    def logits(self, h):
        return self.qwen.lm_head(_rmsnorm(h, self.qwen.model.norm))

    def _rope_mascara(self, L, dispositivo, dtype):
        cos, sin = rope(torch.arange(L, device=dispositivo), self.dim_cabeza, self.theta, dtype)
        G = self.cabezas // self.cabezas_kv
        m = mascara_bloques(L, self.ventana, self.sumideros, G, dispositivo)
        # aditiva (0 / -inf) en el tipo de Qwen, una sola para todas las capas: con una booleana, SDPA arma en cada
        # llamada su propia copia aditiva y cada capa la guarda para la vuelta
        return cos, sin, torch.zeros(m.shape, dtype=dtype, device=dispositivo).masked_fill_(~m, float('-inf'))

    # --- ida ---------------------------------------------------------------------------------------------------

    def ocultos(self, x, sin_asrn=False):
        """x (B, L) piezas -> h (B, L, d) antes de la norma final. sin_asrn=True: Qwen solo (mismas cuentas)."""
        self.revisar_contrato()
        E = self.qwen.model.embed_tokens(x)
        cos, sin, mascara = self._rope_mascara(x.shape[1], x.device, E.dtype)
        recalcular = self.training and torch.is_grad_enabled()
        ck_asrn = recalcular and self.checkpoint in ('asrn', 'todo')
        ck_bloques = recalcular and self.checkpoint == 'todo'
        h, j = E, 0
        for i in range(self.n_capas):
            if ck_bloques and h.requires_grad:
                h = torch.utils.checkpoint.checkpoint(self._bloque, i, h, cos, sin, mascara, use_reentrant=False)
            else:
                h = self._bloque(i, h, cos, sin, mascara)
            if i in self.pos_largo:
                if not sin_asrn:
                    if ck_asrn:
                        h = torch.utils.checkpoint.checkpoint(self._asrn, j, E, h, x, use_reentrant=False)
                    else:
                        h = self._asrn(j, E, h, x)
                j += 1
        return h

    def forward(self, x, sin_asrn=False):
        """Logits (B, L, V) en el tipo de Qwen. Para entrenar usa perdida(): no arma los (B, L, V) logits."""
        return self.logits(self.ocultos(x, sin_asrn))

    def perdida(self, x, y, ignorar=-100, sin_asrn=False):
        """Entropia cruzada media (fp32) por trozos de `trozo_perdida` piezas: la norma final, la salida y la
        entropia de cada trozo se recalculan en la vuelta, asi nunca existen los logits de todo el lote."""
        h = self.ocultos(x, sin_asrn)
        hf, yf = h.reshape(-1, h.shape[-1]), y.reshape(-1)
        recalcular = self.training and torch.is_grad_enabled()

        def trozo(hh, yy):
            return F.cross_entropy(self.logits(hh).float(), yy, ignore_index=ignorar, reduction='sum')

        total = hf.new_zeros((), dtype=torch.float32)
        for s in range(0, hf.shape[0], self.trozo_perdida):
            hh, yy = hf[s:s + self.trozo_perdida], yf[s:s + self.trozo_perdida]
            if recalcular and hh.requires_grad:
                total = total + torch.utils.checkpoint.checkpoint(trozo, hh, yy, use_reentrant=False)
            else:
                total = total + trozo(hh, yy)
        return total / (yf != ignorar).sum().clamp(min=1)


@torch.no_grad()
def verificar_arranque_neutro(modelo, x):
    """True si el injerto no cambia nada: W_o = 0 en todas las capas y los estados finales con y sin ASRN son iguales
    bit a bit. Devuelve (ok, detalle)."""
    estaba = modelo.training
    modelo.eval()
    try:
        wo = [float(c.o.weight.abs().max()) for c in modelo.largos]
        h_con = modelo.ocultos(x)
        h_sin = modelo.ocultos(x, sin_asrn=True)
        igual = torch.equal(h_con, h_sin)
        dif = float((h_con.float() - h_sin.float()).abs().max())
    finally:
        modelo.train(estaba)
    ok = igual and max(wo, default=0.0) == 0.0
    return ok, {'max_abs_W_o': max(wo, default=0.0), 'iguales_bit_a_bit': igual, 'max_dif': dif}


# ---------------------------------------------------------------------------------------------------------------
# Cargar Qwen y guardar / cargar los pesos de ASRN
# ---------------------------------------------------------------------------------------------------------------

def cargar_qwen(nombre='Qwen/Qwen2.5-7B', dispositivo='cuda', dtype=torch.bfloat16):
    """Qwen2ForCausalLM de transformers en `dtype` (bf16), en `dispositivo`, en modo eval y congelado."""
    import transformers
    from transformers import AutoModelForCausalLM
    mayor, menor = (int(p) for p in transformers.__version__.split('.')[:2])
    clave = 'dtype' if (mayor, menor) >= (4, 56) else 'torch_dtype'
    m = AutoModelForCausalLM.from_pretrained(nombre, low_cpu_mem_usage=True, **{clave: dtype})
    m = m.to(dispositivo).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def guardar_asrn(modelo, archivo, **extra):
    """Solo las capas ASRN (fp32) y la configuracion del injerto; Qwen se vuelve a cargar de su repositorio.
    Guardado atomico (asrn.nucleo._guardar_atomico)."""
    pesos = {k: v.detach().cpu() for k, v in modelo.largos.state_dict().items()}
    v89._guardar_atomico({'config': modelo.config, 'largos': pesos, **extra}, archivo)


def cargar_asrn(modelo, archivo):
    """Carga en `modelo` los pesos guardados con guardar_asrn(); devuelve el diccionario (opt, paso, ... si los hay)."""
    datos = torch.load(archivo, map_location='cpu', weights_only=True)
    cfg, mia = datos['config'], modelo.config
    for k in ('d', 'V', 'n_capas', 'largo_cada', 'pos_largo', 'opciones_capa'):
        if cfg.get(k) != mia.get(k):
            raise ValueError(f"{archivo}: {k}={cfg.get(k)!r} no coincide con el modelo ({mia.get(k)!r})")
    modelo.largos.load_state_dict(datos['largos'])
    modelo.revisar_contrato()
    return datos
