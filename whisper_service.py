import whisper


# ==========================================
# DEFAULT WHISPER MODEL
# ==========================================

DEFAULT_MODEL = "small"


# ==========================================
# TRANSCRIBE VIDEO
# ==========================================

def transcribe_audio(audio_path, model_name=DEFAULT_MODEL):

    print("\n================================")
    print("WHISPER TRANSCRIPTION")
    print("================================")

    print(f"Selected Whisper Model: {model_name}")
    print(f"Audio File: {audio_path}")

    # Load selected Whisper model
    print("Loading Whisper model...")

    model = whisper.load_model(model_name)

    print("Whisper model loaded successfully.")

    # Transcribe Hindi speech and translate to English
    result = model.transcribe(
        audio_path,
        language="hi",
        task="translate",
        fp16=False,
        temperature=0
    )

    print("Transcription completed.")

    return result