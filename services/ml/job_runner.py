import argparse
import json
import traceback

try:
    from .bootstrap import ensure_repo_root
except ImportError:
    from services.ml.bootstrap import ensure_repo_root

ensure_repo_root()

from .training import (
    extract_features,
    preprocess_data,
    train_index,
    train_speaker_embedding,
)

JOB_HANDLERS = {
    "preprocess": preprocess_data,
    "extract": extract_features,
    "index": train_index,
    "speaker-embedding": train_speaker_embedding,
}


def _normalize_result(result):
    if isinstance(result, str) and result.lower().startswith("failed"):
        raise RuntimeError(result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run a background ML job")
    parser.add_argument("job_kind", choices=sorted(JOB_HANDLERS.keys()))
    parser.add_argument("payload_path")
    parser.add_argument("result_path")
    args = parser.parse_args()

    with open(args.payload_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    exit_code = 0
    result_payload = {"ok": True, "result": None}
    try:
        result_payload["result"] = _normalize_result(
            JOB_HANDLERS[args.job_kind](**payload)
        )
    except Exception as exc:
        traceback.print_exc()
        result_payload = {"ok": False, "error": str(exc)}
        exit_code = 1

    with open(args.result_path, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, sort_keys=True)

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
