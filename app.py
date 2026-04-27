from flask import (
    Flask, request, send_file, after_this_request, jsonify
)
import subprocess, os, tempfile, traceback
import multiprocessing, json, textwrap, shutil
import requests as http_requests

# ─────────────────────────────────────────────
#  CONFIGURAÇÕES OTIMIZADAS PARA POUCA RAM
# ─────────────────────────────────────────────
app = Flask(__name__)
# Reduzido para 50MB para não estourar a memória RAM da versão gratuita
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Fixado em 1 Thread para evitar que o FFmpeg multiplique processos e consuma toda a RAM
CPU_CORES    = "1" 
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR    = os.path.join(BASE_DIR, "fonts")
os.makedirs(FONTS_DIR, exist_ok=True)

# Procura a fonte em várias localizações; copia para FONTS_DIR se necessário
_FONT_CANDIDATES = [
    (os.path.join(FONTS_DIR, "Autumn_Regular.ttf"), "Autumn"),
    (os.path.join(BASE_DIR,  "Autumn_Regular.ttf"),  "Autumn"),
    (os.path.join(FONTS_DIR, "Anton-Regular.ttf"),   "Anton"),
    (os.path.join(BASE_DIR,  "Anton-Regular.ttf"),   "Anton"),
]
FONT_PATH, FONT_NAME = next(
    ((p, n) for p, n in _FONT_CANDIDATES if os.path.exists(p)),
    ("", "Impact")
)

# Garante que a fonte esteja dentro de FONTS_DIR (para o fontsdir do FFmpeg)
if FONT_PATH and not FONT_PATH.startswith(FONTS_DIR):
    _dest = os.path.join(FONTS_DIR, os.path.basename(FONT_PATH))
    if not os.path.exists(_dest):
        shutil.copy2(FONT_PATH, _dest)
    FONT_PATH = _dest

RESOLUTIONS = {
    "720x1280":  ("720",  "1280"),
    "1080x1080": ("1080", "1080"),
    "1280x720":  ("1280", "720"),
}

# ─────────────────────────────────────────────
#  STATUS (checado pelo frontend ao carregar)
# ─────────────────────────────────────────────
@app.route("/status")
def status():
    return jsonify({
        "groq":      bool(GROQ_API_KEY),
        "font_name": FONT_NAME,
        "font_ok":   bool(FONT_PATH),
    })

# ─────────────────────────────────────────────
#  PÁGINAS
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8").read()

# ─────────────────────────────────────────────
#  PWA
# ─────────────────────────────────────────────
@app.route("/manifest.json")
def manifest():
    return send_file(os.path.join(BASE_DIR, "manifest.json"), mimetype="application/manifest+json")

@app.route("/service-worker.js")
def service_worker():
    resp = send_file(os.path.join(BASE_DIR, "service-worker.js"), mimetype="application/javascript")
    resp.headers["Service-Worker-Allowed"] = "/"
    resp.headers["Cache-Control"] = "no-cache"
    return resp

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_file(os.path.join(BASE_DIR, "static", filename))

# ─────────────────────────────────────────────
#  TRANSCRIÇÃO — GROQ COM TIMESTAMPS POR PALAVRA
# ─────────────────────────────────────────────
def _groq_transcrever(audio_bytes, filename):
    resp = http_requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": (filename, audio_bytes, "audio/mpeg")},
        data={
            "model":                     "whisper-large-v3-turbo",
            "language":                  "pt",
            "response_format":           "verbose_json",
            "timestamp_granularities[]": "word",
        },
        timeout=120
    )
    if resp.status_code != 200:
        raise Exception(f"Groq {resp.status_code}: {resp.text}")

    data  = resp.json()
    texto = data.get("text", "").strip()

    segs = [
        {
            "start": float(s.get("start", 0)),
            "end":   float(s.get("end",   0)),
            "text":  s.get("text", "").strip()
        }
        for s in (data.get("segments") or [])
    ]

    palavras = [
        {
            "word":  w.get("word", "").strip(),
            "start": float(w.get("start", 0)),
            "end":   float(w.get("end",   0)),
        }
        for w in (data.get("words") or [])
        if w.get("word", "").strip()
    ]

    return texto, segs, palavras

@app.route("/transcrever", methods=["POST"])
def transcrever():
    if not GROQ_API_KEY:
        return jsonify({"erro": "GROQ_API_KEY não configurada no servidor."}), 400
    aud = request.files.get("audio")
    if not aud:
        return jsonify({"erro": "Nenhum áudio enviado."}), 400
    try:
        texto, segs, palavras = _groq_transcrever(aud.read(), aud.filename or "audio.mp3")
        return jsonify({"texto": texto, "segmentos": segs, "palavras": palavras})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

# ─────────────────────────────────────────────
#  GERADOR ASS — estilo TikTok karaoke palavra por palavra
# ─────────────────────────────────────────────
def _ts_ass(s: float) -> str:
    h  = int(s // 3600)
    m  = int((s % 3600) // 60)
    sc = int(s % 60)
    cs = int(round((s - int(s)) * 100))
    return f"{h}:{m:02d}:{sc:02d}.{cs:02d}"

PALAVRAS_POR_GRUPO = 4

def gerar_ass(dados: list, w: int, h: int, modo_dados: str = "segmentos") -> str:
    font_size = int(w * 0.074)
    margin_v  = int(h * 0.08)

    c_branco   = "&H00FFFFFF"
    c_apagado  = "&H88AAAAAA"
    c_contorno = "&H00000000"
    c_caixa    = "&HAA000000"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Tron,{FONT_NAME},{font_size},{c_branco},{c_apagado},{c_contorno},{c_caixa},-1,0,0,0,100,100,0,0,3,6,2,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []

    if modo_dados == "palavras" and dados:
        grupos = [
            dados[i : i + PALAVRAS_POR_GRUPO]
            for i in range(0, len(dados), PALAVRAS_POR_GRUPO)
        ]

        for grupo in grupos:
            if not grupo:
                continue
            start = grupo[0]["start"]
            end   = grupo[-1]["end"]
            if end <= start:
                end = start + 0.5

            partes = []
            for palavra in grupo:
                dur_cs = max(1, int(round((palavra["end"] - palavra["start"]) * 100)))
                texto_w = (
                    palavra["word"]
                    .strip()
                    .replace("{", "").replace("}", "")
                    .replace("\\", "")
                )
                partes.append(f"{{\\k{dur_cs}}}{texto_w}")

            texto_linha = " ".join(partes)
            lines.append(
                f"Dialogue: 0,{_ts_ass(start)},{_ts_ass(end)},"
                f"Tron,,0,0,0,,{texto_linha}"
            )

    else:
        for seg in dados:
            txt = seg.get("text", "").strip()
            if not txt:
                continue
            if len(txt) > 35:
                txt = (
                    textwrap.fill(txt, width=35, max_lines=2, placeholder="...")
                    .replace("\n", "\\N")
                )
            txt = txt.replace("{", "").replace("}", "")
            lines.append(
                f"Dialogue: 0,{_ts_ass(seg['start'])},{_ts_ass(seg['end'])},"
                f"Tron,,0,0,0,,{txt}"
            )

    return header + "\n".join(lines)

def _esc(txt: str) -> str:
    return (
        txt.replace("\\", "\\\\")
           .replace("'",  "\\'")
           .replace(":",  "\\:")
           .replace("[",  "\\[")
           .replace("]",  "\\]")
           .replace(",",  "\\,")
    )

def build_vf_estatico(w: str, h: str, legenda: str) -> str:
    scale = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"
    )
    if not legenda.strip():
        return scale

    txt  = _esc(legenda.strip())
    font = f"fontfile={FONT_PATH}" if FONT_PATH else f"font={FONT_NAME}"
    fs   = int(int(w) * 0.074)
    mb   = int(int(h) * 0.06)
    dt   = (
        f"drawtext={font}:text='{txt}':"
        f"fontcolor=white:fontsize={fs}:"
        f"bordercolor=black:borderw=6:"
        f"shadowcolor=black@0.65:shadowx=2:shadowy=3:"
        f"box=1:boxcolor=black@0.40:boxborderw=14:"
        f"x=(w-text_w)/2:y=h-text_h-{mb}"
    )
    return f"{scale},{dt}"

@app.route("/converter", methods=["POST"])
def converter():
    img_file      = request.files.get("imagem")
    aud_file      = request.files.get("audio")
    resolucao     = request.form.get("resolucao",    "1080x1080")
    legenda_txt   = request.form.get("legenda",      "").strip()
    modo_leg      = request.form.get("modo_legenda", "nenhuma")
    palavras_json = request.form.get("palavras",     "")
    segs_json     = request.form.get("segmentos",    "")

    if not img_file or not aud_file:
        return "Imagem e áudio são obrigatórios.", 400

    w_str, h_str = RESOLUTIONS.get(resolucao, ("1080", "1080"))
    w, h = int(w_str), int(h_str)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            img_ext  = os.path.splitext(img_file.filename)[1] or ".jpg"
            aud_ext  = os.path.splitext(aud_file.filename)[1] or ".mp3"
            img
