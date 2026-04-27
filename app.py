from flask import (
    Flask, request, send_file, after_this_request, jsonify
)
import subprocess, os, tempfile, traceback
import multiprocessing, json, textwrap, shutil
import requests as http_requests

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024 # Aumentado para 500MB para áudios longos

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
CPU_CORES    = str(multiprocessing.cpu_count())
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

# Garante que a fonte esteja dentro de FONTS_DIR
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

@app.route("/status")
def status():
    return jsonify({
        "groq":      bool(GROQ_API_KEY),
        "font_name": FONT_NAME,
        "font_ok":   bool(FONT_PATH),
    })

@app.route("/")
def index():
    return open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8").read()

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
#  TRANSCRIÇÃO E PROMPT
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
        timeout=300 # Tempo limite maior para áudios longos
    )
    if resp.status_code != 200:
        raise Exception(f"Groq {resp.status_code}: {resp.text}")

    data  = resp.json()
    texto = data.get("text", "").strip()

    segs = [{"start": float(s.get("start", 0)), "end": float(s.get("end", 0)), "text": s.get("text", "").strip()} for s in (data.get("segments") or [])]
    palavras = [{"word": w.get("word", "").strip(), "start": float(w.get("start", 0)), "end": float(w.get("end", 0))} for w in (data.get("words") or []) if w.get("word", "").strip()]

    return texto, segs, palavras

@app.route("/transcrever", methods=["POST"])
def transcrever():
    if not GROQ_API_KEY:
        return jsonify({"erro": "GROQ_API_KEY não configurada."}), 400
    aud = request.files.get("audio")
    if not aud:
        return jsonify({"erro": "Nenhum áudio enviado."}), 400
    try:
        texto, segs, palavras = _groq_transcrever(aud.read(), aud.filename or "audio.mp3")
        return jsonify({"texto": texto, "segmentos": segs, "palavras": palavras})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route("/gerar-prompt", methods=["POST"])
def gerar_prompt():
    data = request.json
    transcricao = data.get("texto", "").strip()
    if not transcricao:
        return jsonify({"erro": "Texto vazio."}), 400

    try:
        resp = http_requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": "Você é um especialista em criar prompts para geração de imagens com IA. Dado um texto em português, crie um prompt em inglês detalhado e criativo para gerar uma imagem de capa de vídeo. Responda APENAS com o prompt, sem explicações, sem aspas."},
                    {"role": "user", "content": transcricao}
                ],
                "max_tokens": 300,
                "temperature": 0.8
            },
            timeout=30
        )
        if resp.status_code != 200:
            return jsonify({"erro": f"Groq: {resp.text}"}), 500
        prompt = resp.json()["choices"][0]["message"]["content"].strip()
        return jsonify({"prompt": prompt})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

# ─────────────────────────────────────────────
#  GERADOR ASS
# ─────────────────────────────────────────────
def _ts_ass(s: float) -> str:
    h, m, sc, cs = int(s // 3600), int((s % 3600) // 60), int(s % 60), int(round((s - int(s)) * 100))
    return f"{h}:{m:02d}:{sc:02d}.{cs:02d}"

PALAVRAS_POR_GRUPO = 4

def gerar_ass(dados: list, w: int, h: int, modo_dados: str = "segmentos") -> str:
    font_size, margin_v = int(w * 0.074), int(h * 0.08)
    c_branco, c_apagado, c_contorno, c_caixa = "&H00FFFFFF", "&H88AAAAAA", "&H00000000", "&HAA000000"

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
        grupos = [dados[i : i + PALAVRAS_POR_GRUPO] for i in range(0, len(dados), PALAVRAS_POR_GRUPO)]
        for grupo in grupos:
            if not grupo: continue
            start, end = grupo[0]["start"], grupo[-1]["end"]
            if end <= start: end = start + 0.5
            partes = []
            for palavra in grupo:
                dur_cs = max(1, int(round((palavra["end"] - palavra["start"]) * 100)))
                texto_w = palavra["word"].strip().replace("{", "").replace("}", "").replace("\\", "")
                partes.append(f"{{\\k{dur_cs}}}{texto_w}")
            lines.append(f"Dialogue: 0,{_ts_ass(start)},{_ts_ass(end)},Tron,,0,0,0,,{' '.join(partes)}")
    else:
        for seg in dados:
            txt = seg.get("text", "").strip()
            if not txt: continue
            if len(txt) > 35: txt = textwrap.fill(txt, width=35, max_lines=2, placeholder="...").replace("\n", "\\N")
            lines.append(f"Dialogue: 0,{_ts_ass(seg['start'])},{_ts_ass(seg['end'])},Tron,,0,0,0,,{txt.replace('{', '').replace('}', '')}")
    return header + "\n".join(lines)

def _esc(txt: str) -> str:
    return txt.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace("[", "\\[").replace("]", "\\]").replace(",", "\\,")

def build_vf_estatico(w: str, h: str, legenda: str) -> str:
    scale = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"
    if not legenda.strip(): return scale
    txt, font = _esc(legenda.strip()), f"fontfile={FONT_PATH}" if FONT_PATH else f"font={FONT_NAME}"
    fs, mb = int(int(w) * 0.074), int(int(h) * 0.06)
    dt = f"drawtext={font}:text='{txt}':fontcolor=white:fontsize={fs}:bordercolor=black:borderw=6:shadowcolor=black@0.65:shadowx=2:shadowy=3:box=1:boxcolor=black@0.40:boxborderw=14:x=(w-text_w)/2:y=h-text_h-{mb}"
    return f"{scale},{dt}"

# ─────────────────────────────────────────────
#  CONVERSOR PRINCIPAL
# ─────────────────────────────────────────────
@app.route("/converter", methods=["POST"])
def converter():
    img_file, aud_file = request.files.get("imagem"), request.files.get("audio")
    resolucao, legenda_txt = request.form.get("resolucao", "1080x1080"), request.form.get("legenda", "").strip()
    modo_leg, palavras_json, segs_json = request.form.get("modo_legenda", "nenhuma"), request.form.get("palavras", ""), request.form.get("segmentos", "")
    
    if not img_file or not aud_file: 
        return "Imagem e áudio são obrigatórios.", 400
    
    w_str, h_str = RESOLUTIONS.get(resolucao, ("1080", "1080"))
    w, h = int(w_str), int(h_str)

    # Se a pessoa só queria o prompt, não geramos legenda no vídeo
    if modo_leg == "prompt":
        modo_leg = "nenhuma"

    try:
        with tempfile.TemporaryDirectory() as tmp:
            img_path = os.path.join(tmp, "img" + os.path.splitext(img_file.filename)[1])
            aud_path = os.path.join(tmp, "aud" + os.path.splitext(aud_file.filename)[1])
            img_file.save(img_path)
            aud_file.save(aud_path)
            
            fd, out_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            
            scale_vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"
            vf, ass_path = scale_vf, None
            
            if modo_leg == "auto":
                dados_ass, modo_dados = None, "segmentos"
                if palavras_json:
                    try:
                        p = json.loads(palavras_json)
                        if p: dados_ass, modo_dados = p, "palavras"
                    except: pass
                if dados_ass is None and segs_json:
                    try: dados_ass = json.loads(segs_json)
                    except: pass
                if dados_ass:
                    try:
                        ass_content = gerar_ass(dados_ass, w, h, modo_dados)
                        ass_path = os.path.join(tmp, "legenda.ass")
                        with open(ass_path, "w", encoding="utf-8") as f: f.write(ass_content)
                        vf = f"{scale_vf},ass={ass_path}" + (f":fontsdir={FONTS_DIR}" if os.path.isdir(FONTS_DIR) else "")
                    except Exception as err: app.logger.error(f"ASS Error: {err}"); ass_path = None
            
            if ass_path is None and modo_leg == "estatica" and legenda_txt: 
                vf = build_vf_estatico(w_str, h_str, legenda_txt)
            
            # FFMPEG OTIMIZADO PARA ARQUIVOS LONGOS
            cmd = [
                "ffmpeg", "-y", 
                "-loop", "1", 
                "-framerate", "15", # Reduzido para 15fps (acelera demais e economiza RAM)
                "-i", img_path, 
                "-i", aud_path, 
                "-vf", vf, 
                "-map", "0:v", "-map", "1:a", 
                "-c:v", "libx264", 
                "-preset", "ultrafast", 
                "-tune", "stillimage", 
                "-crf", "28", 
                "-max_muxing_queue_size", "1024", # Previne o erro em áudios de várias horas
                "-threads", "0", # Usa 100% da CPU disponível automaticamente
                "-pix_fmt", "yuv420p", 
                "-c:a", "aac", "-b:a", "128k", 
                "-shortest", 
                "-movflags", "+faststart", 
                out_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800) # Timeout estendido para 30min
        
        if result.returncode != 0: 
            return f"FFmpeg Error:\n{result.stderr[-2000:]}", 500
        
        @after_this_request
        def _cleanup(response):
            try: os.unlink(out_path)
            except: pass
            return response
            
        return send_file(out_path, mimetype="video/mp4", as_attachment=True, download_name="tron_clipe.mp4")
    except subprocess.TimeoutExpired:
        return "Tempo limite excedido (30 min).", 504
    except Exception: 
        return f"Error:\n{traceback.format_exc()}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

@app.route("/healthz")
def healthz(): return "OK", 200
        
