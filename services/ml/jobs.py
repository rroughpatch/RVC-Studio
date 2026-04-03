import datetime
import json
import os
import shlex
import subprocess
import threading
import uuid

try:
    from .bootstrap import ensure_repo_root
except ImportError:
    from services.ml.bootstrap import ensure_repo_root

ensure_repo_root()

from lib import BASE_DIR, LOG_DIR, config
from .training import get_training_log_path, start_training_process, write_job_header

JOBS_DIR = os.path.join(LOG_DIR, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _job_dir(job_id: str):
    path = os.path.join(JOBS_DIR, job_id)
    os.makedirs(path, exist_ok=True)
    return path


def _job_snapshot(job_id: str):
    with _LOCK:
        job = _JOBS.get(job_id)
        return None if job is None else dict(job)


def _update_job(job_id: str, **updates):
    with _LOCK:
        _JOBS[job_id].update(updates)
        return dict(_JOBS[job_id])


def _register_job(kind: str, metadata: dict):
    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    job = {
        "id": job_id,
        "kind": kind,
        "status": "queued",
        "created_at": _now(),
        "started_at": None,
        "completed_at": None,
        "pid": None,
        "exit_code": None,
        "command": None,
        "log_path": os.path.join(job_dir, "output.log"),
        "payload_path": os.path.join(job_dir, "payload.json"),
        "result_path": os.path.join(job_dir, "result.json"),
        "metadata": metadata,
        "result": None,
        "error": None,
    }
    with _LOCK:
        _JOBS[job_id] = job
    return dict(job)


def _start_process(job_id: str, cmd: list[str], log_path: str | None = None):
    job = _job_snapshot(job_id)
    if job is None:
        raise KeyError(job_id)

    target_log_path = log_path or job["log_path"]
    command_str = " ".join(shlex.quote(part) for part in cmd)
    write_job_header(
        target_log_path,
        f"starting {job['kind']} job",
        {"job_id": job_id, "command": command_str},
    )
    with open(target_log_path, "a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )

    _update_job(
        job_id,
        status="running",
        started_at=_now(),
        pid=process.pid,
        command=cmd,
        log_path=target_log_path,
    )
    watcher = threading.Thread(
        target=_watch_process,
        args=(job_id, process),
        daemon=True,
        name=f"ml-job-{job_id}",
    )
    watcher.start()
    return _job_snapshot(job_id)


def _watch_process(job_id: str, process: subprocess.Popen):
    return_code = process.wait()
    job = _job_snapshot(job_id)
    if job is None:
        return

    result = None
    error = None
    result_path = job["result_path"]
    if os.path.isfile(result_path):
        try:
            with open(result_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("ok"):
                result = payload.get("result")
            else:
                error = payload.get("error")
        except Exception as exc:
            error = str(exc)

    updates = {
        "completed_at": _now(),
        "exit_code": return_code,
        "result": result,
        "error": error,
    }
    if return_code == 0 and not error:
        updates["status"] = "completed"
    else:
        updates["status"] = "failed"
        updates["error"] = error or f"Process exited with code {return_code}"
    _update_job(job_id, **updates)


def _write_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _submit_runner_job(kind: str, payload: dict):
    job = _register_job(kind, payload)
    _write_json(job["payload_path"], payload)
    return _start_process(
        job["id"],
        [
            config.python_cmd,
            "-m",
            "services.ml.job_runner",
            kind,
            job["payload_path"],
            job["result_path"],
        ],
    )


def submit_preprocess_job(payload: dict):
    return _submit_runner_job("preprocess", payload)


def submit_extract_job(payload: dict):
    return _submit_runner_job("extract", payload)


def submit_index_job(payload: dict):
    return _submit_runner_job("index", payload)


def submit_speaker_embedding_job(payload: dict):
    return _submit_runner_job("speaker-embedding", payload)


def submit_train_job(payload: dict):
    job = _register_job("train", payload)
    try:
        process, cmd, train_log = start_training_process(**payload)
    except Exception as exc:
        write_job_header(job["log_path"], "failed to prepare train job", {"error": exc})
        return _update_job(
            job["id"],
            status="failed",
            completed_at=_now(),
            error=str(exc),
        )

    train_result = {
        "log_path": train_log,
        "model_log_name": f"{payload['exp_dir']}_{payload['version']}_{payload['sr']}",
    }
    _write_json(job["result_path"], {"ok": True, "result": train_result})
    write_job_header(train_log, "training job started", {"job_id": job["id"]})
    _update_job(
        job["id"],
        status="running",
        started_at=_now(),
        pid=process.pid,
        command=cmd,
        log_path=get_training_log_path(
            payload["exp_dir"], payload["version"], payload["sr"]
        ),
        result=train_result,
    )
    watcher = threading.Thread(
        target=_watch_process,
        args=(job["id"], process),
        daemon=True,
        name=f"ml-job-{job['id']}",
    )
    watcher.start()
    return _job_snapshot(job["id"])


def list_jobs():
    with _LOCK:
        return sorted(
            (dict(job) for job in _JOBS.values()),
            key=lambda job: job["created_at"],
            reverse=True,
        )


def get_job(job_id: str):
    return _job_snapshot(job_id)


def get_job_logs(job_id: str, tail: int | None = None):
    job = _job_snapshot(job_id)
    if job is None:
        return None

    log_path = job["log_path"]
    content = ""
    if os.path.isfile(log_path):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
        if tail is not None:
            lines = lines[-tail:]
        content = "".join(lines)
    return {"job_id": job_id, "log_path": log_path, "content": content}
