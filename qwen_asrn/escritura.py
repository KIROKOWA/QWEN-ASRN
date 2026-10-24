"""Escritura O(1) por pieza de QwenASRN: cache KV de Qwen en anillo (ventana W + S sumideros) acoplada a la escritura
O(1) de asrn (_Escritor): anillos de casilleros (T x 2^n_bits x chunk posiciones por capa), _MemoriaHawkes por clave
y la racha.

Por pieza nueva, en cada bloque de Qwen: 1 pregunta contra W + 8 llaves (fijo). En cada capa ASRN: T x chunk
candidatos (los mismos `chunk` companeros de casillero mas recientes que la ida sobre el texto entero), Hawkes por
diccionario y racha de un numero. Nada recorre el texto.

Memoria por texto: la cache KV y los anillos son fijos; lo que crece es la memoria de cada posicion que una pregunta
puede leer (phi, ids, Kc, sufijo, reloj fp64, log alfa). asrn._Escritor guarda ademas V = v(E_{s+1}) (d numeros fp32
por posicion y por capa: con d = 3584, 14 KB). Aqui, con v_factorizado=True (por omision), no: como v es lineal y sin
sesgo,
    sum_k att_k v(E_{s_k + 1}) = v(sum_k att_k E_{s_k + 1}),
asi que se guarda el texto, se toman los T*C vectores de pieza de los candidatos de la tabla de embeddings de Qwen y
se aplica v una sola vez. Mismas cuentas salvo el redondeo, 26x menos memoria por pieza (presupuesto.py).

memoria_de_lo_ajeno=True (como asrn.escribir): ASRN copia solo de la entrada, no de lo que el modelo escribio.
"""
import argparse

import torch
import torch.nn.functional as F

from asrn import nucleo as v89
from asrn.nucleo import _anillo_desde, _anillo_meter, _anteriores, _claves_de, _MemoriaHawkes, firmas_de_piezas

from .modelo import RANURAS_SUMIDERO, _rmsnorm, _rotar, _SwiGLU, rope


class EscritorQwenASRN:
    """inicio(x0) -> logits de la ultima pieza; paso(pieza) -> logits de la siguiente. precision_asrn: 'fp32' (como
    asrn._Escritor) o 'bf16' (autocast, como al entrenar). prefill_max: la entrada se procesa entera de una vez hasta
    esas piezas y el resto pieza por pieza (misma funcion, memoria acotada para entradas enormes)."""

    def __init__(self, modelo, memoria_de_lo_ajeno=True, v_factorizado=True, precision_asrn='fp32'):
        self.m, self.ajeno, self.fact, self.precision = modelo, memoria_de_lo_ajeno, v_factorizado, precision_asrn

    # --- memoria reservada ---------------------------------------------------------------------------------------

    @staticmethod
    def _reservar(x, cap):
        buf = x.new_zeros((x.shape[0], cap) + tuple(x.shape[2:]))
        buf[:, :x.shape[1]] = x
        return buf

    def _crecer(self):
        cap = 2 * self.cap
        self.texto, self.sig = self._reservar(self.texto, cap), self._reservar(self.sig, cap)
        for c in self.cache:
            for n in c:
                c[n] = self._reservar(c[n], cap)
        self.cap = cap

    def bytes_estado(self):
        """Bytes en el dispositivo del estado de escritura (cache KV, anillos, memoria de posiciones, Hawkes)."""
        total = sum(t.numel() * t.element_size() for kv in self.kv for t in kv.values())
        total += sum(a.numel() * a.element_size() + c.numel() * c.element_size() for a, c in self.anillos)
        total += sum(t.numel() * t.element_size() for c in self.cache for t in c.values())
        total += sum(h.Bu.numel() * 8 + h.A.numel() * 4 for h in self.hawkes)
        return total + self.texto.numel() * 8 + self.sig.numel() * 4

    # --- Qwen: un bloque con la cache KV en anillo -------------------------------------------------------------------

    def _cache_kv(self, k, v):
        """Cache de un bloque desde k, v (B, Hk, L, hd) del texto entero: anillo de W y sumideros."""
        m = self.m
        B, Hk, L, hd = k.shape
        W = m.ventana
        kv = {n: k.new_zeros(B, Hk, W, hd) for n in ('k', 'v')}
        kv.update({n: k.new_zeros(B, Hk, RANURAS_SUMIDERO, hd) for n in ('ks', 'vs')})
        desde = max(0, L - W)
        ranuras = torch.arange(desde, L, device=k.device) % W
        kv['k'][:, :, ranuras], kv['v'][:, :, ranuras] = k[:, :, desde:], v[:, :, desde:]
        n_s = min(m.sumideros, L)
        kv['ks'][:, :, :n_s], kv['vs'][:, :, :n_s] = k[:, :, :n_s], v[:, :, :n_s]
        return kv

    def _bloque_paso(self, i, h, cos, sin, t):
        """Bloque i de Qwen para la pieza t (h (B, 1, d)): la mete en el anillo y atiende a la ventana y los
        sumideros. Las mismas llaves que mascara_bloques() da a la posicion t en la ida."""
        m = self.m
        c = m.qwen.model.layers[i]
        a = c.self_attn
        B = h.shape[0]
        W, Hk, G = m.ventana, m.cabezas_kv, m.cabezas // m.cabezas_kv
        x = _rmsnorm(h, c.input_layernorm)
        q = a.q_proj(x).view(B, 1, m.cabezas, m.dim_cabeza).transpose(1, 2)
        k = a.k_proj(x).view(B, 1, Hk, m.dim_cabeza).transpose(1, 2)
        v = a.v_proj(x).view(B, 1, Hk, m.dim_cabeza).transpose(1, 2)
        q, k = _rotar(q, cos, sin), _rotar(k, cos, sin)
        kv = self.kv[i]
        kv['k'][:, :, t % W], kv['v'][:, :, t % W] = k[:, :, 0], v[:, :, 0]
        if t < m.sumideros:
            kv['ks'][:, :, t], kv['vs'][:, :, t] = k[:, :, 0], v[:, :, 0]
        dv = h.device
        s = torch.arange(RANURAS_SUMIDERO, device=dv)
        mascara = torch.cat([(s < m.sumideros) & (s <= t - W), torch.arange(W, device=dv) <= t]).view(1, 1, 1, -1)
        llaves = torch.cat([kv['ks'], kv['k']], 2)                                   # (B, Hk, 8 + W, hd)
        valores = torch.cat([kv['vs'], kv['v']], 2)
        o = F.scaled_dot_product_attention(q.view(B, Hk, G, m.dim_cabeza), llaves, valores, attn_mask=mascara)
        h = h + a.o_proj(o.reshape(B, 1, -1))
        mlp = c.mlp
        x = _rmsnorm(h, c.post_attention_layernorm)
        return h + mlp.down_proj(_SwiGLU.apply(mlp.gate_proj(x), mlp.up_proj(x)))

    # --- ASRN: candidatos --------------------------------------------------------------------------------------------

    def _candidatos(self, c, anillo, ids_t):
        """Como asrn.nucleo._candidatos; con v_factorizado, 'V' son los vectores de pieza E_{s+1} de cada candidato s
        (la capa aplica v despues: _atender_cand suma att * V)."""
        B, T = ids_t.shape
        dv = ids_t.device
        pos = anillo[torch.arange(B, device=dv).unsqueeze(1), torch.arange(T, device=dv).unsqueeze(0),
                     ids_t.long()].long()                                            # (B, T, C)
        valido = pos >= 0
        pos = pos.clamp(min=0)
        b = torch.arange(B, device=dv).view(B, 1, 1)
        cand = {n: c[n][b, pos] for n in ('phi', 'Kc', 'Pf', 'B', 'log_alfa')}
        if self.fact:
            cand['V'] = self.m.qwen.model.embed_tokens(self.texto[b, pos + 1]).float()   # (B, T, C, d)
        else:
            cand['V'] = c['V'][b, pos]
        return cand, valido

    # --- entrada -----------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def inicio(self, x0, capacidad=0, prefill_max=None):
        m = self.m
        m.revisar_contrato()
        B, L0 = x0.shape
        assert L0 >= 1, "hace falta al menos una pieza de inicio"
        P = L0 if prefill_max is None else max(1, min(L0, prefill_max))
        dv = m.qwen.model.embed_tokens.weight.device
        x0 = x0.to(dv)
        self.L0, self.cap = L0, max(L0 + 1, capacidad)
        self.texto = self._reservar(x0.long(), self.cap)
        x = x0[:, :P]
        sig = firmas_de_piezas(x)
        self.sig = self._reservar(sig, self.cap)
        sig_cpu = firmas_de_piezas(x.cpu())
        n_max = max(c.vistos for c in m.largos)
        self.ventana_sig = F.pad(sig_cpu, (n_max, 0), value=float('nan'))[:, -n_max:].clone()   # (B, n_max)
        E = m.qwen.model.embed_tokens(x)
        E32 = E.float()
        cos, sin, mascara = m._rope_mascara(P, dv, E.dtype)
        self.kv, self.cache, self.anillos, self.hawkes, self.racha, self.c_cpu = [], [], [], [], [], []
        h, j = E, 0
        for i in range(m.n_capas):
            h, k, v = m._bloque(i, h, cos, sin, mascara, kv=True)
            self.kv.append(self._cache_kv(k, v))
            if i in m.pos_largo:
                capa = m.largos[j]
                with m.autocast_asrn(h.device, self.precision):
                    h32 = h.float()
                    phi, ids, V, Qc, Kc, beta, log_alfa = capa._caracteristicas(E32, h32)
                    Bt = beta.cumsum(1)
                    rep = capa._repeticion(sig, beta, Bt, log_alfa)
                    guardar = [('phi', phi), ('ids', ids), ('Kc', Kc), ('Pf', _anteriores(sig, capa.sufijo)),
                               ('B', Bt), ('log_alfa', log_alfa)]
                    if not self.fact:
                        guardar.append(('V', V))
                    self.cache.append({n: self._reservar(t, self.cap) for n, t in guardar})
                    self.racha.append(rep['racha'][:, -1])
                    n_anillo = min(P, L0 - 1) if self.ajeno else P
                    self.anillos.append(_anillo_desde(ids, n_anillo, capa.chunk, 2 ** capa.n_bits))
                    self.c_cpu.append(capa.c_vistos.cpu())
                    mem = _MemoriaHawkes(B, capa.vistos, dv, Bt.dtype)
                    mem.cargar(_claves_de(sig_cpu, self.c_cpu[-1], capa.vistos), rep['hawkes'], Bt, log_alfa)
                    self.hawkes.append(mem)
                    h = capa(E32, h32, x).to(h.dtype)
                j += 1
        self.e_ult = E32[:, -1:]
        self.t = P
        lg = m.logits(h[:, -1:])[:, -1].float()
        for t in range(P, L0):                                   # el resto de la entrada, pieza por pieza
            lg = self.paso(x0[:, t])
        return lg

    # --- una pieza ---------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def paso(self, pieza):
        m = self.m
        if self.t >= self.cap:
            self._crecer()
        t = self.t
        dv = self.texto.device
        pieza_cpu = pieza.reshape(-1).cpu()
        pieza = pieza.reshape(-1).to(dv)
        self.texto[:, t] = pieza
        self.sig[:, t] = firmas_de_piezas(pieza)
        self.ventana_sig = torch.cat([self.ventana_sig[:, 1:], firmas_de_piezas(pieza_cpu).unsqueeze(1)], 1)
        e = m.qwen.model.embed_tokens(pieza.view(-1, 1))                              # (B, 1, d)
        e32 = e.float()
        cos, sin = rope(torch.tensor([t], device=dv), m.dim_cabeza, m.theta, e.dtype)
        h, j = e, 0
        for i in range(m.n_capas):
            h = self._bloque_paso(i, h, cos, sin, t)
            if i in m.pos_largo:
                capa, c, anillo = m.largos[j], self.cache[j], self.anillos[j]
                with m.autocast_asrn(h.device, self.precision):
                    h32 = h.float()
                    if not self.fact:
                        c['V'][:, t - 1] = capa.v(e32[:, 0])
                    phi_t = capa._phi(torch.cat([self.e_ult, e32], 1))[:, -1:]              # (B, 1, T*nb)
                    ids_t = capa._ids(phi_t)                                                # (B, 1, T)
                    Qc_t, Kc_t, beta_t, log_alfa_t = capa._contexto(h32)                    # (B, 1, .)
                    c['B'][:, t] = c['B'][:, t - 1] + beta_t[:, 0]                          # el reloj avanza
                    c['log_alfa'][:, t] = log_alfa_t[:, 0]
                    c['Pf'][:, t] = _anteriores(self.sig[:, max(0, t - capa.sufijo):t + 1], capa.sufijo)[:, -1]
                    cand, valido = self._candidatos(c, anillo[0], ids_t[:, 0])
                    acc = capa._atender_cand(phi_t, Qc_t, c['Pf'][:, t:t + 1], c['B'][:, t:t + 1], cand, valido)
                    if self.fact:
                        acc['salida'] = capa.v(acc['salida'])
                    claves = _claves_de(self.ventana_sig[:, -capa.vistos:], self.c_cpu[j], capa.vistos)[:, -1]
                    acc.update(capa._repeticion_o1(self.hawkes[j], claves, c['B'][:, t], log_alfa_t[:, 0],
                                                   beta_t[:, 0], self.racha[j]))
                    self.racha[j] = acc['racha'][:, 0]
                    h = capa._combinar(acc, h32).to(h.dtype)
                c['phi'][:, t], c['ids'][:, t], c['Kc'][:, t] = phi_t[:, 0], ids_t[:, 0], Kc_t[:, 0]
                if not self.ajeno or t < self.L0 - 1:                  # t es llave desde la pieza t+1
                    _anillo_meter(*anillo, ids_t[:, 0], t)
                j += 1
        self.e_ult = e32
        self.t += 1
        return m.logits(h)[:, -1].float()


@torch.no_grad()
def escribir(modelo, inicio, n_nuevos=200, temperatura=0.8, top_k=40, top_p=1.0, semilla=0, memoria_de_lo_ajeno=True,
             parar_en=None, conservar_cita=True, prefill_max=None, v_factorizado=True, precision_asrn='fp32'):
    """Escribe n_nuevos piezas despues de `inicio` (B, L) o (L,). Muestreo como asrn.escribir (top_k, top_p,
    conservar_cita). Devuelve (B, L + escritas) en la CPU."""
    estaba = modelo.training
    modelo.eval()
    try:
        x = inicio if inicio.dim() == 2 else inicio.unsqueeze(0)
        if int(x.min()) < 0 or int(x.max()) >= modelo.V:
            raise ValueError(f"inicio: piezas fuera de [0, {modelo.V})")
        g = torch.Generator(device='cpu').manual_seed(semilla)
        esc = EscritorQwenASRN(modelo, memoria_de_lo_ajeno, v_factorizado, precision_asrn)
        lg = esc.inicio(x, capacidad=x.shape[1] + n_nuevos, prefill_max=prefill_max)
        dv = lg.device
        cita_ng = v89._Ngramas(x.cpu().tolist(), modelo.largos[0].vistos) if conservar_cita else None
        salida = [x.cpu()]
        terminado = torch.zeros(x.shape[0], dtype=torch.bool)
        for k in range(n_nuevos):
            cita = cita_ng.siguientes(modelo.V, dv) if cita_ng else None
            sig = v89._muestrear(lg[:, :modelo.V], temperatura, top_k, g, top_p, cita).cpu()       # (B, 1)
            if parar_en is not None:
                sig = torch.where(terminado.unsqueeze(1), torch.full_like(sig, parar_en), sig)
                terminado |= sig[:, 0] == parar_en
            salida.append(sig)
            if cita_ng:
                cita_ng.agregar(sig[:, 0].tolist())
            if (parar_en is not None and bool(terminado.all())) or k == n_nuevos - 1:
                break
            lg = esc.paso(sig[:, 0])
        return torch.cat(salida, 1)
    finally:
        modelo.train(estaba)


def principal(args=None):
    from transformers import AutoTokenizer

    from asrn import Protecciones

    from .modelo import QwenASRN, cargar_asrn, cargar_qwen
    a = argparse.ArgumentParser(description="Escribir con Qwen 2.5 + ASRN (escritura O(1) por pieza)")
    a.add_argument('--modelo', default='Qwen/Qwen2.5-7B')
    a.add_argument('--asrn', required=True, help='pesos ASRN guardados por qwen_asrn.entrenar')
    a.add_argument('--texto', required=True)
    a.add_argument('--n', type=int, default=200)
    a.add_argument('--temperatura', type=float, default=0.8)
    a.add_argument('--top_k', type=int, default=40)
    a.add_argument('--top_p', type=float, default=1.0)
    a.add_argument('--prefill_max', type=int, default=16384)
    a.add_argument('--copiar_lo_propio', action='store_true', help='ASRN copia tambien de lo que escribe')
    a.add_argument('--dispositivo', default='cuda')
    a = a.parse_args(args)
    tok = AutoTokenizer.from_pretrained(a.modelo)
    datos = torch.load(a.asrn, map_location='cpu', weights_only=True)['config']
    modelo = QwenASRN(cargar_qwen(a.modelo, a.dispositivo), ventana=datos['ventana'], sumideros=datos['sumideros'],
                      largo_cada=datos['largo_cada'], opciones_capa=datos['opciones_capa'],
                      protecciones=Protecciones(**datos['protecciones']))
    cargar_asrn(modelo, a.asrn)
    x = torch.tensor([tok.encode(a.texto)])
    y = escribir(modelo, x, a.n, a.temperatura, a.top_k, a.top_p, memoria_de_lo_ajeno=not a.copiar_lo_propio,
                 parar_en=tok.eos_token_id, prefill_max=a.prefill_max)
    print(tok.decode(y[0, x.shape[1]:].tolist()))


if __name__ == '__main__':
    principal()
