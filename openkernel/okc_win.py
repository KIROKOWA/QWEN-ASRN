"""Un okc minimo para Windows (y Colab): el reto solo publica okc para macOS y Linux, asi que este usa su API HTTP
(https://openkernelchallenge.dev/llms.txt). Solo la biblioteca estandar de Python.

    python okc_win.py login                     # GitHub (codigo de dispositivo) y aceptar las reglas: lo hace una persona
    python okc_win.py check submission.py       # la corrida completa, fuera del tablero
    python okc_win.py submit submission.py      # la que cuenta
    python okc_win.py status <job>              # el resultado de una corrida
    python okc_win.py problems                  # limites de hoy y lo que te queda
    python okc_win.py leaderboard [problema]
    python okc_win.py token                     # imprime tu token (para OKC_TOKEN en los Secretos de Colab)

El token se lee de OKC_TOKEN o del archivo ~/.okc_win_token (que escribe login). No lo subas a ningun repositorio."""

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://openkernelchallenge.dev/api/v1"
ARCHIVO_TOKEN = os.path.join(os.path.expanduser("~"), ".okc_win_token")
FINALES = ("done", "failed", "rejected", "duplicate")


def _token():
    tok = os.environ.get("OKC_TOKEN")
    if not tok and os.path.exists(ARCHIVO_TOKEN):
        tok = open(ARCHIVO_TOKEN).read().strip()
    if not tok:
        sys.exit("No hay token: corre `python okc_win.py login` o pon OKC_TOKEN.")
    return tok


def _api(metodo, ruta, cuerpo=None, token=None):
    datos = None if cuerpo is None else json.dumps(cuerpo).encode()
    pedido = urllib.request.Request(BASE + ruta, data=datos, method=metodo)
    pedido.add_header("Content-Type", "application/json")
    pedido.add_header("User-Agent", "okc-win/0.1")
    if token:
        pedido.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(pedido, timeout=60) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"{metodo} {ruta}: HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}")


def login():
    inicio = _api("POST", "/auth/device/start", {})
    print(f"1. Abre {inicio['verification_uri']} y escribe el codigo: {inicio['user_code']}")
    print("   (esperando a que lo autorices en GitHub...)")
    handle = inicio["handle"]
    while True:
        r = _api("POST", "/auth/device/poll", {"handle": handle})
        if r.get("token"):
            break
        handle = r.get("handle", handle)
        time.sleep(max(1, int(inicio.get("interval", 5))))
    token = r["token"]
    with open(ARCHIVO_TOKEN, "w") as f:
        f.write(token)
    print(f"   listo: sesion iniciada; token guardado en {ARCHIVO_TOKEN}")

    yo = _api("GET", "/me", token=token)
    reglas = _api("GET", "/rules")
    if (yo.get("rules") or {}).get("accepted_at"):
        print("Ya habias aceptado las reglas vigentes.")
        return
    print(f"\n2. Antes de enviar hay que aceptar las reglas ({reglas['rules']}) y los terminos ({reglas['terms']}),")
    print(f"   version {reglas['version']}. Leelos. Al enviar, tu codigo queda bajo la licencia Apache 2.0.")
    if input("   Escribe SI para aceptarlos: ").strip().upper() not in ("SI", "SÍ", "YES"):
        sys.exit("No se aceptaron; no podras enviar hasta hacerlo.")
    correo = input("   ¿Puede tu correo ir a Meta? (si/no): ").strip().lower() in ("si", "sí", "yes", "s", "y")
    _api("POST", "/rules/accept", {"version": reglas["version"], "meta_email": correo}, token=token)
    print("   reglas aceptadas.")


def _problema_de(codigo):
    primera = codigo.splitlines()[0] if codigo else ""
    if "problem=" not in primera:
        sys.exit("La primera linea del archivo debe ser `# okc: problem=<problema>`.")
    return primera.split("problem=", 1)[1].strip()


def enviar(archivo, modo):
    codigo = open(archivo, encoding="utf-8").read()
    problema = _problema_de(codigo)
    r = _api("POST", f"/problems/{problema}/submissions", {"code": codigo, "mode": modo}, token=_token())
    print(f"{problema}: corrida {r['job_id']} ({modo}); esperando al juez...")
    status(r["job_id"], carpeta=os.path.dirname(os.path.abspath(archivo)))


def status(job, carpeta="."):
    token = _token()
    visto = None
    while True:
        r = _api("GET", f"/jobs/{job}", token=token)
        estado = r.get("status")
        if estado != visto:
            print(f"  {estado}")
            visto = estado
        if estado in FINALES:
            break
        time.sleep(10)
    os.makedirs(os.path.join(carpeta, "profiles"), exist_ok=True)
    salida = os.path.join(carpeta, "profiles", f"submission.{job}.json")
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1, ensure_ascii=False)
    resultado = {k: v for k, v in r.items() if k not in ("log_tail", "code", "profile")}
    print(json.dumps(resultado, indent=1, ensure_ascii=False)[:6000])
    if r.get("log_tail"):
        print("--- lo que imprimio la corrida (final) ---\n" + r["log_tail"][-3000:])
    print(f"(todo el resultado en {salida})")


def main(args):
    if not args:
        sys.exit(__doc__)
    orden = args[0]
    if orden == "login":
        login()
    elif orden in ("check", "submit") and len(args) == 2:
        enviar(args[1], "check" if orden == "check" else "run")
    elif orden == "status" and len(args) == 2:
        status(args[1])
    elif orden == "problems":
        print(json.dumps(_api("GET", "/problems", token=os.environ.get("OKC_TOKEN") or (
            open(ARCHIVO_TOKEN).read().strip() if os.path.exists(ARCHIVO_TOKEN) else None)), indent=1,
            ensure_ascii=False))
    elif orden == "leaderboard":
        ruta = f"/leaderboard/{args[1]}" if len(args) > 1 else "/leaderboard"
        filas = _api("GET", ruta).get("rows", [])
        for i, fila in enumerate(filas[:20], 1):
            print(f"{i:3d}  {fila.get('team', '?'):24s}  {fila.get('speedup', '')}")
    elif orden == "token":
        print(_token())
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
