# Evaluating the Accuracy of Speech to Text Technologies in Turkish-Accented English
# Presented at the 19th Student Conference of Linguistics, İstanbul University, Türkiye, 3 April 2026
# Nuran Orhan & Ali Çağan Kaya

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from faster_whisper import WhisperModel

AUDIO_FILES = [
    # "/path/to/a.wav",
]
AUDIO_DIR = "/Users/alicagank/Desktop/orhan-kaya/stimuli"   # in_path
OUTDIR    = "/Users/alicagank/Desktop/orhan-kaya/output"    # out_path

# Faster-Whisper settings
MODEL_SIZE         = "base"        # base
DEVICE             = "cpu"         # "cpu" or "cuda"
COMPUTE_TYPE       = "int8"        # CPU default
FORCE_LANGUAGE     = "en"
TASK               = "transcribe"  # "transcribe" or "translate"
USE_VAD            = True          # built-in VAD filter
VAD_PARAMETERS     = {"min_silence_duration_ms": 500}
WORD_TIMESTAMPS    = False         # no stamps

SUPPORTED = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".webm", ".mp4"}

def _to_srt_timestamp(t: float) -> str:
    ms = int(round(float(t) * 1000))
    h = ms // 3_600_000; ms %= 3_600_000
    m = ms // 60_000;    ms %= 60_000
    s = ms // 1000;      ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _write_txt(path: Path, text: str):
    path.write_text((text or "").strip() + "\n", encoding="utf-8")

def _write_srt(path: Path, segments_list):
    lines = []
    for i, seg in enumerate(segments_list, 1):
        start = _to_srt_timestamp(seg["start"])
        end   = _to_srt_timestamp(seg["end"])
        text  = seg["text"].strip()
        lines += [str(i), f"{start} --> {end}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")

def _write_segments_csv(path: Path, segments_list):
    # segments_list
    rows = []
    for i, seg in enumerate(segments_list):
        base = {
            "segment_index": i,
            "start_sec": float(seg["start"]),
            "end_sec": float(seg["end"]),
            "text": seg["text"].strip(),
        }
        rows.append(base)

        # If word timestamps requested, also write a separate *_words.csv
        if WORD_TIMESTAMPS and seg.get("words"):
            pass

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")

def _write_words_csv(path: Path, segments_list):
    word_rows = []
    for i, seg in enumerate(segments_list):
        for w in seg.get("words", []) or []:
            word_rows.append({
                "segment_index": i,
                "word": w["word"],
                "start_sec": float(w["start"]),
                "end_sec": float(w["end"])
            })
    if word_rows:
        pd.DataFrame(word_rows).to_csv(path, index=False, encoding="utf-8")

def _gather_from_dir(d: str):
    p = Path(d)
    if not p.is_dir():
        return []
    return sorted([q for q in p.rglob("*") if q.suffix.lower() in SUPPORTED])

def _collect_files():
    files = []
    if AUDIO_DIR:
        files.extend(_gather_from_dir(AUDIO_DIR))
    files.extend([Path(f) for f in AUDIO_FILES])
    files = [f for f in files if f.is_file() and f.suffix.lower() in SUPPORTED]
    # de-dup
    uniq = []
    seen = set()
    for f in files:
        s = str(f.resolve())
        if s not in seen:
            uniq.append(Path(s))
            seen.add(s)
    return uniq

def transcribe_audio_files(files, outdir=None):
    if not files:
        print("No valid audio files found.")
        return

    print(f"Loading faster-whisper model: {MODEL_SIZE} (device={DEVICE}, compute_type={COMPUTE_TYPE})")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    for audio in tqdm(files, desc="Transcribing"):
        out_base = (Path(outdir) if outdir else audio.parent)
        out_base.mkdir(parents=True, exist_ok=True)
        stem = audio.stem

        segments_iter, info = model.transcribe(
            str(audio),
            language=FORCE_LANGUAGE,      # None for auto normalde EN
            task=TASK,                    # "transcribe" or "translate"
            vad_filter=USE_VAD,
            vad_parameters=VAD_PARAMETERS,
            word_timestamps=WORD_TIMESTAMPS
        )

        # SRT/CSV list creation
        segments_list = []
        full_text_parts = []
        for seg in segments_iter:
            entry = {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text or ""
            }
            if WORD_TIMESTAMPS and seg.words:
                entry["words"] = [{"word": w.word, "start": float(w.start), "end": float(w.end)}
                                  for w in seg.words]
            segments_list.append(entry)
            full_text_parts.append(entry["text"])

        text = "".join(full_text_parts).strip()

        # Write outputs
        _write_txt(out_base / f"{stem}.txt", text)
        _write_srt(out_base / f"{stem}.srt", segments_list)
        _write_segments_csv(out_base / f"{stem}_segments.csv", segments_list)
        if WORD_TIMESTAMPS:
            _write_words_csv(out_base / f"{stem}_words.csv", segments_list)

    print("Done.")

if __name__ == "__main__":
    files = _collect_files()
    transcribe_audio_files(files, outdir=OUTDIR or None)
