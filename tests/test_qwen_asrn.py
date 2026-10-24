"""Pruebas en CPU con un Qwen2 diminuto (misma arquitectura que Qwen 2.5, pesos al azar): python -m pytest tests"""
import dataclasses

import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip('transformers')

from asrn import CapaASRNEstable  # noqa: E402

from qwen_asrn import presupuesto  # noqa: E402
from qwen_asrn.escritura import EscritorQwenASRN  # noqa: E402
from qwen_asrn.modelo import (QwenASRN, _atencion_bloques, mascara_bloques, rope, verificar_arranque_neutro)  # noqa


def qwen_chico(dtype=torch.float32, capas=6, semilla=0):
    torch.manual_seed(semilla)
    cfg = transformers.Qwen2Config(vocab_size=300, hidden_size=64, intermediate_size=176, num_hidden_layers=capas,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                   rope_theta=1e6, tie_word_embeddings=False, attn_implementation='eager')
    m = transformers.Qwen2ForCausalLM(cfg).eval()
    with torch.no_grad():                      # sesgos de q, k, v distintos de cero, como en Qwen 2.5
        for c in m.model.layers:
            for lin in (c.self_attn.q_proj, c.self_attn.k_proj, c.self_attn.v_proj):
                lin.bias.normal_(std=0.1)
    return m.to(dtype)


def textos(B, L, V=300, semilla=1):
    g = torch.Generator().manual_seed(semilla)
    base = torch.randint(0, V, (B, L // 3), generator=g)
    return torch.cat([base, base, torch.randint(0, V, (B, L - 2 * (L // 3)), generator=g)], 1)   # con repeticiones


def despertar(modelo, semilla=2):
    """Pesos de ASRN al azar (W_o y la puerta distintos de cero) para probar el camino de ASRN, no solo el neutro."""
    g = torch.Generator().manual_seed(semilla)
    with torch.no_grad():
        for c in modelo.largos:
            c.o.weight.copy_(torch.randn(c.o.weight.shape, generator=g) * 0.05)
            c.g.weight.copy_(torch.randn(c.g.weight.shape, generator=g) * 0.05)
            c.cq.weight.copy_(torch.randn(c.cq.weight.shape, generator=g) * 0.1)
            c.w_reloj.copy_(torch.randn(c.w_reloj.shape, generator=g) * 0.1)
            c.u_hawkes.copy_(torch.randn(c.u_hawkes.shape, generator=g))
            c.gamma.copy_(torch.rand(c.gamma.shape, generator=g))
    return modelo


OPC = dict(n_bits=4, chunk=4, confirmar=4, afin_oculto=32)       # casilleros chicos: muchos candidatos en un texto corto


# --- la ida de Qwen ----------------------------------------------------------------------------------------------

def test_qwen_propio_igual_que_transformers_dentro_de_la_ventana():
    q = qwen_chico()
    m = QwenASRN(q, ventana=64, contrato='libre', opciones_capa=OPC)
    x = textos(2, 40)
    with torch.no_grad():
        ref = q(x).logits
        mio = m(x, sin_asrn=True)
    assert (ref - mio).abs().max() < 1e-4


def test_atencion_por_bloques_igual_que_mascara_densa():
    torch.manual_seed(0)
    B, Hq, Hk, L, hd, W, S = 2, 4, 2, 53, 8, 8, 3
    q, k, v = torch.randn(B, Hq, L, hd), torch.randn(B, Hk, L, hd), torch.randn(B, Hk, L, hd)
    o = _atencion_bloques(q, k, v, W, S, mascara_bloques(L, W, S, Hq // Hk, 'cpu'))
    p = torch.arange(L).view(-1, 1)
    s = torch.arange(L).view(1, -1)
    densa = (s <= p) & ((s > p - W) | (s < S))
    ref = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), attn_mask=densa)
    assert (o - ref).abs().max() < 1e-5


# --- contrato y arranque neutro -------------------------------------------------------------------------------------

def test_arranque_neutro_bit_a_bit_en_bf16():
    m = QwenASRN(qwen_chico(torch.bfloat16), ventana=16, opciones_capa=OPC)      # contrato estricto: Qwen bf16
    assert m.pos_largo == [0, 2, 4]
    x = textos(2, 50)
    ok, det = verificar_arranque_neutro(m, x)
    assert ok, det
    with torch.no_grad():
        assert torch.equal(m(x), m(x, sin_asrn=True))


def test_contrato_de_precision():
    with pytest.raises(ValueError):
        QwenASRN(qwen_chico(torch.float32), opciones_capa=OPC)                   # estricto: Qwen debe ir en bf16
    m = QwenASRN(qwen_chico(torch.bfloat16), ventana=16, opciones_capa=OPC)
    assert all(p.dtype == torch.float32 for p in m.largos.parameters())
    assert all(isinstance(c, CapaASRNEstable) for c in m.largos)
    m.largos[1].to(torch.bfloat16)
    with pytest.raises(ValueError):
        m(textos(1, 10))


def test_solo_asrn_recibe_gradiente_y_W_o_aprende_en_el_paso_0():
    m = QwenASRN(qwen_chico(torch.bfloat16), ventana=16, opciones_capa=OPC).train()
    x = textos(2, 40)
    m.perdida(x[:, :-1], x[:, 1:]).backward()
    assert all(p.grad is None for p in m.qwen.parameters())
    for c in m.largos:
        assert float(c.o.weight.grad.abs().max()) > 0          # W_o recibe gradiente aunque sea 0
        assert float(c.v.weight.grad.abs().max()) == 0         # el resto, recien desde el paso 1 (como LoRA)


@pytest.mark.parametrize('ck', ['asrn', 'todo'])
def test_checkpoint_mismas_cuentas(ck):
    x = textos(2, 37)
    resultados = {}
    for modo in ('no', ck):
        m = despertar(QwenASRN(qwen_chico(), ventana=8, sumideros=2, checkpoint=modo, contrato='libre',
                               trozo_perdida=16, precision_asrn='fp32', opciones_capa=OPC)).train()
        p = m.perdida(x[:, :-1], x[:, 1:])
        p.backward()
        resultados[modo] = (p.detach(), [q.grad.clone() for q in m.parametros_entrenables()])
    (p0, g0), (p1, g1) = resultados['no'], resultados[ck]
    assert torch.allclose(p0, p1, atol=1e-6)
    assert all(torch.allclose(a, b, atol=1e-6, rtol=1e-4) for a, b in zip(g0, g1))


def test_rmsnorm_y_swiglu_propios_dan_el_gradiente_de_autograd():
    from qwen_asrn.modelo import _RMSNorm, _SwiGLU
    torch.manual_seed(0)
    x = torch.randn(3, 5, 16, dtype=torch.float64, requires_grad=True)
    w = torch.randn(16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, w: _RMSNorm.apply(x, w, 1e-6), (x, w))
    a, b = (torch.randn(4, 7, dtype=torch.float64, requires_grad=True) for _ in range(2))
    assert torch.autograd.gradcheck(_SwiGLU.apply, (a, b))


# --- escritura O(1) --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('fact', [True, False])
def test_escritura_o1_igual_que_la_ida_entera(fact):
    m = despertar(QwenASRN(qwen_chico(), ventana=8, sumideros=2, contrato='libre', precision_asrn='fp32',
                           opciones_capa=OPC)).eval()
    x = textos(2, 60)
    L0 = 20
    with torch.no_grad():
        ref = m(x).float()[:, L0 - 1:]
    esc = EscritorQwenASRN(m, memoria_de_lo_ajeno=False, v_factorizado=fact)
    sal = torch.stack([esc.inicio(x[:, :L0])] + [esc.paso(x[:, t]) for t in range(L0, x.shape[1])], 1)
    assert (sal[:, :-1] - ref[:, :-1]).abs().max() < 2e-4
    with torch.no_grad():                        # y de verdad usa ASRN: sin ASRN los logits son otros
        assert (m(x, sin_asrn=True).float()[:, L0 - 1:] - ref).abs().max() > 1e-2


def test_prefill_por_pasos_y_v_factorizado_dan_lo_mismo():
    m = despertar(QwenASRN(qwen_chico(), ventana=8, sumideros=2, contrato='libre', precision_asrn='fp32',
                           opciones_capa=OPC)).eval()
    x = textos(2, 50)
    L0 = 30
    salidas = {}
    for ajeno in (True, False):
        for fact in (True, False):
            for pre in (None, 7):
                esc = EscritorQwenASRN(m, memoria_de_lo_ajeno=ajeno, v_factorizado=fact)
                salidas[ajeno, fact, pre] = torch.stack(
                    [esc.inicio(x[:, :L0], prefill_max=pre)] + [esc.paso(x[:, t]) for t in range(L0, 50)], 1)
        base = salidas[ajeno, True, None]
        for k, v in salidas.items():
            if k[0] == ajeno:
                assert (v - base).abs().max() < 2e-4, k
    assert (salidas[True, True, None] - salidas[False, True, None]).abs().max() > 1e-4   # lo ajeno cambia algo


def test_escribir_corre_en_bf16():
    from qwen_asrn import escribir
    m = despertar(QwenASRN(qwen_chico(torch.bfloat16), ventana=8, sumideros=2, opciones_capa=OPC))
    y = escribir(m, textos(2, 20), n_nuevos=15)
    assert y.shape == (2, 35) and int(y.max()) < 300


def test_rope_fp64_igual_que_qwen():
    q = qwen_chico()
    pos = torch.arange(100)
    cos, sin = rope(pos, 16, 1e6, torch.float32)
    c_ref, s_ref = q.model.rotary_emb(torch.zeros(1, 100, 16), pos.unsqueeze(0))
    assert (cos - c_ref[0]).abs().max() < 1e-5 and (sin - s_ref[0]).abs().max() < 1e-5


# --- presupuesto --------------------------------------------------------------------------------------------------

def test_presupuesto_cuenta_los_parametros_de_verdad():
    with torch.device('meta'):
        capa = CapaASRNEstable(3584)
    assert sum(p.numel() for p in capa.parameters()) == presupuesto.parametros_capa_asrn(
        3584, presupuesto.ConfigInjerto())['total'] == 26_518_306
    assert presupuesto.parametros_qwen(presupuesto.ConfigQwen())['total'] == 7_615_616_512
    q = qwen_chico()
    cq = presupuesto.ConfigQwen(capas=6, d=64, ff=176, cabezas=4, cabezas_kv=2, dim_cabeza=16, V=300)
    assert presupuesto.parametros_qwen(cq)['total'] == sum(p.numel() for p in q.parameters())
    m = QwenASRN(q, contrato='libre', opciones_capa=OPC)
    inj = dataclasses.replace(presupuesto.ConfigInjerto(), **OPC)
    assert sum(p.numel() for p in m.largos.parameters()) == 3 * presupuesto.parametros_capa_asrn(64, inj)['total']


def test_presupuesto_cuenta_lo_que_guarda_autograd():
    """Bytes guardados para la vuelta por pieza (sin la parte de ASRN, que es su checkpoint) contra lo que autograd
    guarda de verdad, contando cada almacenamiento una vez."""
    q = qwen_chico(torch.bfloat16, capas=4)
    W, L = 8, 32
    m = QwenASRN(q, ventana=W, sumideros=2, checkpoint='asrn', opciones_capa=OPC).train()
    vistos = {}

    def guardar(t):
        if t.device.type != 'meta':
            st = t.untyped_storage()
            vistos[st.data_ptr()] = st.nbytes()
        return t

    pesos = {p.untyped_storage().data_ptr() for p in m.parameters()} | {b.untyped_storage().data_ptr()
                                                                          for b in m.buffers()}
    x = textos(1, L + 1)
    with torch.autograd.graph.saved_tensors_hooks(guardar, lambda t: t):
        p = m.perdida(x[:, :-1], x[:, 1:])
    total = sum(n for ptr, n in vistos.items() if ptr not in pesos)
    cq = presupuesto.ConfigQwen(capas=4, d=64, ff=176, cabezas=4, cabezas_kv=2, dim_cabeza=16, V=300)
    inj = dataclasses.replace(presupuesto.ConfigInjerto(ventana=W, sumideros=2), **OPC)
    esperado = presupuesto.activaciones_por_pieza(cq, inj, 'asrn')['total'] * L
    p.backward()
    # tolerancia: cos/sin de RoPE (L x hd, compartidos por las capas) y los tensores chicos de la perdida
    assert abs(total - esperado) / esperado < 0.05, (total, esperado)
