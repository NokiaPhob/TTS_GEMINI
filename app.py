"""
Gemini 3.8 Flash TTS - Voice Replication Studio (Streamlit)

Flow:
  1. Paste Gemini API key
  2. Record/upload a reference clip (10-30 s) + a consent clip (same speaker)
  3. Create replicated voice  -> POST /v1beta/voices  (client.voices.create)
  4. Type a script, optional style -> client.interactions.create (gemini-3.8-flash-tts)
  5. Play + download the WAV

Run:  streamlit run app.py
Needs: google-genai>=2.25.0, streamlit>=1.39, pydub (+ ffmpeg for non-WAV uploads)
"""

import base64
import io

import streamlit as st
from google import genai
from pydub import AudioSegment

MODEL = "gemini-3.8-flash-tts"
MIN_SEC, MAX_SEC = 10.0, 30.0  # reference clip length required by the API docs

# Verbatim consent statements from the official voice replication docs.
CONSENT_PHRASES = {
    "Thai (th-TH)": "ฉันเป็นเจ้าของเสียงนี้ และฉันยินยอมให้ Google ใช้เสียงนี้เพื่อสร้างแบบจำลองเสียงสังเคราะห์",
    "English (en-US)": "I am the owner of this voice and I consent to Google using this voice to create a synthetic voice model.",
    "Chinese Simplified (zh-CN)": "我是此声音的拥有者并授权谷歌使用此声音创建语音合成模型",
    "Japanese (ja-JP)": "私はこの音声の所有者であり、Googleがこの音声を使用して音声合成モデルを作成することを承認します。",
    "Korean (ko-KR)": "나는 이 음성의 소유자이며 구글이 이 음성을 사용하여 음성 합성 모델을 생성할 것을 허용합니다.",
    "Vietnamese (vi-VN)": "Tôi là chủ sở hữu giọng nói này và tôi đồng ý cho Google sử dụng giọng nói này để tạo mô hình giọng nói tổng hợp.",
    "Indonesian (id-ID)": "Saya pemilik suara ini dan saya menyetujui Google menggunakan suara ini untuk membuat model suara sintetis.",
    "Hindi (hi-IN)": "मैं इस आवाज का मालिक हूं और मैं सिंथेटिक आवाज मॉडल बनाने के लिए Google को इस आवाज का उपयोग करने की सहमति देता हूं",
    "French (fr-FR)": "Je suis le propriétaire de cette voix et j'autorise Google à utiliser cette voix pour créer un modèle de voix synthétique.",
    "German (de-DE)": "Ich bin der Eigentümer dieser Stimme und bin damit einverstanden, dass Google diese Stimme zur Erstellung eines synthetischen Stimmmodells verwendet.",
    "Spanish (es-ES)": "Soy el propietario de esta voz y doy mi consentimiento para que Google la utilice para crear un modelo de voz sintética.",
    "Portuguese Brazil (pt-BR)": "Eu sou o proprietário desta voz e autorizo o Google a usá-la para criar um modelo de voz sintética.",
}


# ----------------------------------------------------------------- helpers
def normalize_audio(raw: bytes) -> tuple[bytes, float]:
    """Convert any supported audio to 24 kHz mono 16-bit WAV (recommended by docs)."""
    seg = AudioSegment.from_file(io.BytesIO(raw))
    seg = seg.set_frame_rate(24000).set_channels(1).set_sample_width(2)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return buf.getvalue(), len(seg) / 1000.0


def audio_picker(label: str, key: str) -> bytes | None:
    """Record-or-upload widget. Returns raw audio bytes or None."""
    tab_rec, tab_up = st.tabs(["🎙️ Record", "📁 Upload"])
    with tab_rec:
        rec = st.audio_input(f"Record: {label}", key=f"{key}_rec")
    with tab_up:
        up = st.file_uploader(
            f"Upload: {label}",
            type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
            key=f"{key}_up",
        )
    if rec is not None:
        return rec.getvalue()
    if up is not None:
        return up.getvalue()
    return None


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


# -------------------------------------------------------------------- page
st.set_page_config(page_title="Gemini 3.8 Flash TTS - Voice Replication", page_icon="🎙️")
st.title("🎙️ Gemini 3.8 Flash TTS")
st.caption("Voice replication + expressive text-to-speech")

with st.sidebar:
    api_key = st.text_input("Gemini API Key", type="password", help="Get one at aistudio.google.com/apikey")
    st.divider()
    st.markdown(
        "**Use responsibly.** Only replicate your own voice, or a voice whose owner "
        "has given you clear permission."
    )

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to begin.")
    st.stop()

client = genai.Client(api_key=api_key)
st.session_state.setdefault("voice_ref", None)
st.session_state.setdefault("audio_out", None)

# ---------------------------------------------------- step 1: replicate voice
st.header("1. Replicate a voice")
st.markdown(
    f"The API needs **two clips from the same adult speaker**, recorded on the same "
    f"mic in the same room:\n"
    f"- **Reference clip**: {MIN_SEC:.0f}-{MAX_SEC:.0f} s of clean, natural speech\n"
    f"- **Consent clip**: the speaker reading the consent statement below"
)

st.subheader("Reference clip")
source_raw = audio_picker("reference voice sample", "src")

st.subheader("Consent clip")
lang = st.selectbox("Consent language", list(CONSENT_PHRASES), index=0)
st.code(CONSENT_PHRASES[lang], language=None)
st.caption("Read this exactly, clearly, in the same voice and setup as the reference clip.")
consent_raw = audio_picker("consent statement", "consent")

col_a, col_b = st.columns(2)
display_name = col_a.text_input("Voice name", value="My Replicated Voice")
stateless = col_b.checkbox(
    "Stateless (don't store on Google's side)",
    help="Returns a voicekey_... you keep client-side. Expires after 7 days. "
    "Unchecked = stored voice_... ID, kept 1 year from last use.",
)

if st.button("Create replicated voice", type="primary"):
    if not source_raw or not consent_raw:
        st.error("Please provide both the reference clip and the consent clip.")
    else:
        try:
            src_wav, src_sec = normalize_audio(source_raw)
            con_wav, _ = normalize_audio(consent_raw)
            if not (MIN_SEC <= src_sec <= MAX_SEC):
                st.error(
                    f"Reference clip is {src_sec:.1f} s. It must be {MIN_SEC:.0f}-{MAX_SEC:.0f} s."
                )
            else:
                with st.spinner("Creating voice..."):
                    voice = {
                        "model": MODEL,
                        "type": "replicated",
                        "replicated": {
                            "source_audio": {"mime_type": "audio/wav", "data": b64(src_wav)},
                            "consent_audio": {"mime_type": "audio/wav", "data": b64(con_wav)},
                        },
                    }
                    if not stateless:
                        voice["display_name"] = display_name
                    result = client.voices.create(store=not stateless, voice=voice)
                ref = result.key if stateless else result.id
                st.session_state.voice_ref = ref
                st.success("Voice created.")
                st.code(ref, language=None)
                st.caption("Save this ID/key if you want to reuse the voice later.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Voice creation failed: {e}")
            st.caption(
                "Common causes: consent clip doesn't match the statement, speaker differs "
                "between clips, background noise, or non-WAV decoding needing ffmpeg."
            )

with st.expander("Use an existing voice instead"):
    manual = st.text_input("Paste a voice_... ID or voicekey_...")
    if st.button("Use this voice") and manual.strip():
        st.session_state.voice_ref = manual.strip()
    if st.button("List my stored replicated voices"):
        try:
            resp = client.voices.list(type_=["replicated"])
            voices = resp.voices or []
            if not voices:
                st.write("No stored replicated voices found.")
            for v in voices:
                st.code(f"{v.id}  |  {v.display_name}", language=None)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not list voices: {e}")

# ------------------------------------------------------ step 2: generate speech
st.header("2. Generate speech")
if st.session_state.voice_ref:
    st.caption(f"Active voice: `{st.session_state.voice_ref[:24]}...`")
else:
    st.warning("Create or select a voice above first.")

script = st.text_area(
    "Script",
    height=160,
    placeholder="Type exactly what the voice should say...",
    help="Spoken verbatim. Keep delivery notes out of the text; use the Style field instead. "
    "Max ~8,192 input tokens.",
)
style = st.text_input(
    "Style (optional)",
    placeholder="e.g. warm and conversational",
    help="Turn-level delivery (emotion, pace). Leave empty for the most natural, stable result.",
)
st.caption(
    "Inline tags for momentary sounds: `<laugh>` `<sigh>` `<breath>` `<short pause>` `<long pause>`. "
    "Use CAPS for emphasis. Keep tags in English even for Thai text."
)

if st.button("Generate speech", type="primary", disabled=not st.session_state.voice_ref):
    if not script.strip():
        st.error("Please enter a script.")
    else:
        try:
            block = {"type": "text", "text": script.strip()}
            if style.strip():
                block["annotations"] = [{"type": "speech_metadata", "style": style.strip()}]
            with st.spinner("Generating..."):
                interaction = client.interactions.create(
                    model=MODEL,
                    input=[{"type": "user_input", "content": [block]}],
                    response_format={"type": "audio"},  # unary default = WAV with RIFF header
                    generation_config={"speech_config": [{"voice": st.session_state.voice_ref}]},
                )
            # Unary output is already a complete .wav, so no manual header needed.
            st.session_state.audio_out = base64.b64decode(interaction.output_audio.data)
        except Exception as e:  # noqa: BLE001
            st.session_state.audio_out = None
            st.error(f"Generation failed: {e}")

if st.session_state.audio_out:
    st.audio(st.session_state.audio_out, format="audio/wav")
    st.download_button(
        "⬇️ Download WAV",
        data=st.session_state.audio_out,
        file_name="replicated_speech.wav",
        mime="audio/wav",
    )
