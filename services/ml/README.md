# ML Service

`services/ml` is now the canonical home for the Python compute layer.

The repo root still exports compatibility wrappers (`api.py`, `server/*`, and the
old top-level ML module filenames) so the existing Streamlit and CLI flows keep
working while the rest of the migration continues.

The repo root remains the shared data root for:

- `models`
- `songs`
- `logs`
- `output`
- `datasets`
- `configs`

Canonical ML modules:

- `api.py`
- `server/`
- `training_cli.py`
- `tts_cli.py`
- `uvr5_cli.py`
- `preprocessing_utils.py`
- `vc_infer_pipeline.py`
- `pitch_extraction.py`
- `rvc_for_realtime.py`

Canonical start commands:

```bash
uv run python -m services.ml
uv run python -m services.ml.api
uv run python -m services.ml.training_cli --help
```

The JavaScript workspace uses Vite+ (`vp`) for orchestration, but the ML runtime
still runs through the Python environment managed at the repo root.
