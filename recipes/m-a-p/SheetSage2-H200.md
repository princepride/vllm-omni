# SheetSage2 audio-to-score preprocessing — H200

## Summary

- Vendor: m-a-p
- Model: `m-a-p/SheetSage2` with the `m-a-p/MERT-v2-FullSong` parent
- Task: recording → ABC score, MIDI and timed music annotations
- Mode: standalone official Transformers inference; optional YuE2 request export
- Hardware: one NVIDIA H200 (141 GB)

## When to use this recipe

Use a recording as the musical score for a YuE2 cover, or inspect its melodies,
chords and structure. The tool calls SheetSage2's official `transcribe()` API.
It does not register SheetSage2 as a native vLLM model or start an Omni server.

The transcription tool works independently of YuE2. Sending its generated
request requires the YuE2 adapter from
[PR #7886](https://github.com/vllm-project/vllm-omni/pull/7886), which is not
part of this change. Until that PR is merged, use a separate serving checkout
containing it. An unmodified main checkout cannot serve YuE2 yet.

## Supported model contract

| Item | Contract |
| --- | --- |
| Input | One local recording decodable by FFmpeg; upstream converts it to mono, 24 kHz |
| Duration | Entire recording by default; `--max-seconds` limits preprocessing to a prefix |
| Full score | Default: melody voices and chords; exported request uses `cot=full` |
| Melody score | `--melody-only`: vocal and instrumental melodies, without score/playback chords; request uses `cot=melody` |
| Outputs | `score.abc`, `transcription.mid`, upstream annotations and `manifest.json` |
| YuE2 handoff | `--lyrics-file` and `--style` together also write `yue2_request.json` |
| Output directory | Must be new or empty; failed exports produce no YuE2 request |
| Deployment profile | One process, one H200, FP32 weight loading and BF16 inference autocast |

`--melody-only` does not isolate or clone a singer. The upstream model still
decodes its task annotations and retains those raw annotation files. YuE2
uses the exported score to condition newly generated audio; a transcription
does not guarantee an exact reproduction of the recording.

## References

- [Official SheetSage2 model card](https://huggingface.co/m-a-p/SheetSage2)
- [Pinned SheetSage2 source](https://huggingface.co/m-a-p/SheetSage2/tree/cafc0df1021e14f49e928c4b345f5959d414ef64)
- [Official YuE2 cover documentation](https://huggingface.co/m-a-p/YuE2-3B#%EF%B8%8F-cover)
- Tool: [`tools/sheetsage2_transcribe.py`](../../tools/sheetsage2_transcribe.py)

## Hardware

- Accelerator: one NVIDIA H200, 141 GB; no multi-device interconnect required.
- Qualification: local M4A transcription, melody-only prefix and full-score
  whole-song export. Other devices, CPU inference and performance scaling are
  not qualified by this recipe.

## Software environment

- Ubuntu 22.04, Python 3.10, NVIDIA driver 590.48.01.
- PyTorch/torchaudio 2.8.0 (CUDA 12.8 wheels), Transformers 4.45.2.
- FFmpeg 4.4.2 was used for the local M4A check; upstream recommends FFmpeg 6.1.
- vLLM: not required in the preprocessing environment.
- vLLM-Omni: tool developed against main `a038b38179e9c788d3af6a8e73f94652afb247e4`.

## Command

Run from this repository's root. Create a **separate environment**: the
official transcriber's Transformers version differs from Omni's serving
dependencies. FFmpeg must be on `PATH`.

```bash
uv venv --python 3.10 .venv-sheetsage2
uv pip install --python .venv-sheetsage2/bin/python \
  -r tools/requirements/sheetsage2.txt

.venv-sheetsage2/bin/python tools/sheetsage2_transcribe.py reference.wav \
  --trust-remote-code --device cuda:0 --melody-only \
  --output-dir outputs/reference-score
```

The default Hub checkpoint and its custom Python code are pinned to
`cafc0df1021e14f49e928c4b345f5959d414ef64`. Its configuration pins the MERT-v2
parent to `d8ba1c745e733b3908ce6ad16ebeb17ac7600a42`. Review that code before
passing `--trust-remote-code`. `--revision` can override the default pin;
custom Hub IDs use their default revision unless one is supplied.

For offline use, download **both** model snapshots, including their Python,
JSON and safetensors files, before disconnecting:

```bash
.venv-sheetsage2/bin/hf download m-a-p/SheetSage2 \
  --revision cafc0df1021e14f49e928c4b345f5959d414ef64 \
  --local-dir models/SheetSage2
.venv-sheetsage2/bin/hf download m-a-p/MERT-v2-FullSong \
  --revision d8ba1c745e733b3908ce6ad16ebeb17ac7600a42 \
  --local-dir models/MERT-v2-FullSong

.venv-sheetsage2/bin/python tools/sheetsage2_transcribe.py reference.wav \
  --model models/SheetSage2 --base-model-path models/MERT-v2-FullSong \
  --local-files-only --trust-remote-code --device cuda:0 \
  --melody-only --max-seconds 45 \
  --lyrics-file lyrics.txt --style 'Chinese folk, guzheng, gentle vocals' \
  --seed 831001 --output-dir outputs/cover-score
```

Supply target lyrics in `lyrics.txt`, including YuE2 section tags such as
`[Verse]` and `[Chorus]`. Match the lyric phrasing to the reference melody.
Remove `--max-seconds 45` for the whole recording. Remove `--melody-only` to
retain chords and export a `cot=full` request. Use a fresh output directory
for each run.

Start YuE2 in its **own Omni environment**, with a checkout containing #7886:

```bash
vllm serve m-a-p/YuE2-3B --omni --port 8091
```

Submit the file from the preprocessing checkout (adjust its path if the
serving checkout is elsewhere):

```bash
curl --fail-with-body http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-binary @outputs/cover-score/yue2_request.json \
  --output cover.wav
```

The JSON contains the score itself, not a server-local file path. It sets
`stream=false`, requests WAV, and preserves an optional generation seed.
Use `--yue2-model` to match a custom served model name. The tool does not
submit the request or set a generation token budget; the server's own
duration and context limits apply.

## Verification

After a successful transcription:

```bash
.venv-sheetsage2/bin/python - <<'PY'
import json
from pathlib import Path
import mido

output = Path("outputs/cover-score")
abc = (output / "score.abc").read_text(encoding="utf-8")
request = json.loads((output / "yue2_request.json").read_text(encoding="utf-8"))
assert abc.strip() and request["extra_params"]["abc"] == abc
assert request["extra_params"]["cot"] == "melody"
midi = mido.MidiFile(output / "transcription.mid")
assert any(msg.type == "note_on" and msg.velocity for track in midi.tracks for msg in track)
print("ABC, MIDI and YuE2 request verified")
PY
```

Run the dependency-free CLI contract tests in the repository's normal test
environment (with pytest and pytest-mock installed):

```bash
python -m pytest tests/tools/test_sheetsage2_transcribe.py -m 'core_model and cpu' -q
```

These tests cover request construction, loading options, validation and
failure handling with a mocked backend. They do not measure transcription
accuracy or generated cover similarity.

Local H200 checks with the pinned backend:

| Input / mode | Result |
| --- | --- |
| 45 s M4A prefix, melody-only | 579 ABC characters; parseable MIDI with 118 note-on events |
| 242.219 s M4A, full score | 3,033 ABC characters; parseable MIDI with 1,087 note-on events |
| Default Hub ID, 10 s prefix | Successful ABC and MIDI export |
| Local snapshots, fresh Transformers module cache | Successful offline full-score export |
| Both generated JSON requests, #7886 adapter at `5cc8acc942e76fa6f12f8b2f2558c2663abc2aa5` | Validation and prompt construction passed (919 / 2,821 prompt tokens) |

The adapter check used vLLM 0.30.0 in a separate serving environment. It
checked the real request schema, tokenizer and context budget; it did not
run HTTP synthesis or evaluate the resulting cover's audio quality. The
local recording and generated scores are not included in the repository.

## Notes

- Weights load in FP32 because upstream merges MERT adapters before inference
  autocast. `--dtype fp32` disables BF16 autocast.
- For local snapshots the tool prepares transitive Python dependencies in
  Transformers' module cache. This avoids the pinned Transformers version's
  direct-import-only copy behavior without editing upstream files.
- SheetSage2 manages long recordings with its own overlapping windows. This
  tool leaves windowing and notation generation to the official implementation.
- Failed runs may retain upstream diagnostic annotations. Choose a new output
  directory when retrying; no incomplete request is published for synthesis.
- Model weights have upstream license terms (CC BY-NC 4.0); this repository
  does not bundle weights or recordings.

## Supported features

| Feature | Status |
| --- | --- |
| Audio-to-score preprocessing | Official backend; local file input |
| YuE2 score-conditioned generation | Exported request; serving requires #7886 |
| Native Omni inference, continuous batching, TP/PP/SP | Not implemented for SheetSage2 |
| Streaming transcription or cover audio | Not supported by this tool |
