import whisper

print("Loading Whisper model...")

model = whisper.load_model("small")

print("Whisper model loaded successfully!")


def transcribe_audio(audio_path):

    print("Transcribing:", audio_path)

    result = model.transcribe(
    audio_path,
    language="hi",
    task="translate",
    fp16=False,
    temperature=0
)
    segments = []

    for segment in result["segments"]:
        segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip()
        })

    return segments