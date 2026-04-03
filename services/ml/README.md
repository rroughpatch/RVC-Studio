# ML Service

`services/ml` is the target home for the Python compute layer.

The FastAPI entrypoint now lives at `services/ml/api.py`, and the `server`
package has been copied under `services/ml/server`. The repo root still exports
compatibility wrappers (`api.py` and `server/*`) so the existing Streamlit and
CLI flows keep working while the rest of the migration continues.

The Python code has not been moved into this directory yet because the current app
still derives `BASE_DIR` from the process working directory in `lib/__init__.py`.
Moving the runtime without adding compatibility shims would immediately break path
resolution for `models`, `songs`, `logs`, `output`, `datasets`, and `configs`.

The first migration targets for this service are:

- `api.py`
- `server/`
- `training_cli.py`
- `tts_cli.py`
- `uvr5_cli.py`
- `vc_infer_pipeline.py`
- `rvc_for_realtime.py`
- `preprocessing_utils.py`
- `pitch_extraction.py`

During this transition, the repo root remains the data root.

The JavaScript workspace now uses Vite+ (`vp`) for orchestration, but the Python
runtime still needs explicit path and process-boundary changes before it can move
under `services/ml` safely.
