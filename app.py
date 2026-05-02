from flask import Flask, request, jsonify, send_from_directory
import uuid, subprocess, os
from vieneu.standard import VieNeuTTS
import torch

# ====================== CẤU HÌNH ======================
BASE_REPO = "pnnbao-ump/VieNeu-TTS-0.3B"
LORA_REPO = "pnnbao-ump/VieNeu-TTS-0.3B-lora-ngoc-huyen"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Device: {device.upper()}")

print("⏳ Đang tải Base Model + Codec...")
tts = VieNeuTTS(
    backbone_repo=BASE_REPO,
    backbone_device=device,
    codec_repo="neuphonic/distill-neucodec",   # Codec nhẹ và ổn định hơn cho 0.3B
    codec_device=device,
    hf_token=None   # Để trống vì repo public
)

print(f"🎤 Đang load LoRA Ngọc Huyền...")
tts.load_lora_adapter(LORA_REPO)

print("✅ Load thành công giọng Ngọc Huyền!")
# ====================== CẤU HÌNH ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

PUBLIC_URL = "http://127.0.0.1:5000"

app = Flask(__name__)

# ====================== VOICE ======================
VOICE_MAP = {
    "ngoc_huyen": os.path.join(BASE_DIR, "ref_ngoc_huyen.wav"),
    "default": None
}

# ====================== API ======================
@app.route('/tts', methods=['POST'])
def synthesize():
    try:
        text     = request.form.get('text', '').strip()
        voice    = request.form.get('voice', 'default')
        rate     = request.form.get('rate', '0')
        language = request.form.get('language', 'vi')

        if not text:
            return jsonify({'code': 1, 'msg': 'text is empty', 'data': ''})

        # ====================== RATE → TEMPERATURE ======================
        temperature = 0.35
        try:
            r = float(str(rate).replace('%', '').replace('+', '').replace('-', ''))
            if rate and '-' in str(rate):
                temperature = 0.30 + (r / 100.0) * 0.1
            else:
                temperature = 0.40 - (r / 100.0) * 0.1
            temperature = max(0.25, min(0.65, temperature))
        except:
            temperature = 0.35

        # ====================== VOICE ======================
        ref_codes = None
        if voice in VOICE_MAP and VOICE_MAP[voice]:
            ref_path = VOICE_MAP[voice]
            if os.path.exists(ref_path):
                try:
                    ref_codes = tts.encode_reference(ref_path)
                except:
                    ref_codes = None

        # ====================== INFERENCE ======================
        if ref_codes is not None:
            audio = tts.infer(
                text=text,
                voice=ref_codes,
                temperature=temperature
            )
        else:
            audio = tts.infer(
                text=text,
                temperature=temperature
            )

        # ====================== SAVE ======================
        uid = uuid.uuid4().hex
        wav_path = os.path.join(AUDIO_DIR, f'{uid}.wav')
        mp3_path = os.path.join(AUDIO_DIR, f'{uid}.mp3')

        tts.save(audio, wav_path)

        # Convert WAV → MP3 (cần ffmpeg cài local)
        subprocess.run([
            'ffmpeg', '-i', wav_path,
            '-af', 'volume=2.0',
            '-q:a', '2', '-y', mp3_path
        ], capture_output=True)

        if os.path.exists(wav_path):
            os.remove(wav_path)

        url = f'{PUBLIC_URL}/audio/{uid}.mp3'

        return jsonify({
            'code': 0,
            'msg': 'success',
            'data': url,
            'temperature': round(temperature, 3)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'code': 2, 'msg': str(e), 'data': ''})


@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'VieNeu-TTS local'})


# ====================== RUN LOCAL ======================
if __name__ == "__main__":
    print("\n🚀 Server chạy local!")
    print(f"👉 http://127.0.0.1:5000/tts")
    app.run(host='0.0.0.0', port=5000, debug=False)