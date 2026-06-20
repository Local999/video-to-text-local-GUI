# Quick Start — Local Video/Audio Transcription

Everything is installed and verified on this machine. To transcribe a video:

## Easiest: the `transcribe` wrapper

```bash
./transcribe /path/to/your/video.mp4
```

- Copies the file into `videos/` (or `audios/` for `.mp3`/`.m4a`), runs Whisper,
  and writes the transcript to `transcripts/<name>.txt`.
- Defaults to **English**. For another language: `./transcribe file.mp4 --language ru`
- To transcribe everything already sitting in `videos/`: just run `./transcribe`
- Extra flags pass straight through, e.g. `./transcribe clip.mp4 --cleanup`

## Manual (equivalent)

```bash
source venv/bin/activate
# drop files into videos/ then:
python main.py --type video --language en
# or, for audio files placed in audios/:
python main.py --type audio --language en
```

## Where things are

- **Transcripts:** `transcripts/<name>.txt`
- **Logs:** `logs/transcriber.log`
- **Model:** `configurations/params.yaml` → `transcription_model` (currently `large-v3`).
  Options: `tiny | base | small | medium | large-v3` — bigger = more accurate but slower on CPU.

## Web GUI (optional)

Prefer a browser? `pip install -r requirements-gui.txt` then `python app.py`
and open http://127.0.0.1:7860 — upload, pick a model, transcribe, download.

## Notes

- This Mac has no NVIDIA GPU, so transcription runs on **CPU**. The first run of a given
  model downloads its weights once (`large-v3` ≈ 3 GB), cached afterward under `~/.cache/whisper`.
- `ffmpeg` is provided by the pip-bundled **imageio-ffmpeg** (symlinked at `venv/bin/ffmpeg`),
  so no system/Homebrew install is needed. The `transcribe` wrapper puts it on PATH automatically.
- Optional `--cleanup` (needs a local **Ollama** server) and `--diarize` (needs a HuggingFace
  token in a `.env` file) are supported but not configured yet — see `README.md`.
