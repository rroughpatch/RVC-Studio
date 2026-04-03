import argparse
import os
import subprocess
try:
    from .bootstrap import ensure_repo_root
except ImportError:
    from services.ml.bootstrap import ensure_repo_root

ensure_repo_root()

from fastapi import FastAPI, HTTPException
from lib.audio import audio2bytes
from .jobs import (
    get_job,
    get_job_logs,
    list_jobs,
    submit_extract_job,
    submit_index_job,
    submit_preprocess_job,
    submit_speaker_embedding_job,
    submit_train_job,
)
from .server import STATUS
from .server.rvc import convert_vocals, list_rvc_models
from .server.types import (
    ExtractFeaturesJobParams,
    PreprocessJobParams,
    RVCInferenceParams,
    SpeakerEmbeddingJobParams,
    TTSInferenceParams,
    TrainIndexJobParams,
    TrainJobParams,
    UVRInferenceParams,
)
from .server.uvr import list_uvr_denoise_models, list_uvr_models, split_vocals
from .tts_cli import TTS_MODELS_DIR, generate_speech
from lib.utils import get_optimal_threads, gc_collect
from lib import BASE_DIR, config

server = FastAPI()
TTS_METHODS = ["edge", "speecht5", "bark", "tacotron2", "vits", "silero"]


def _model_data(model):
    model_dump = getattr(model, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return model.dict()


def _list_tts_speakers():
    embeddings_dir = os.path.join(TTS_MODELS_DIR, "embeddings")
    if not os.path.isdir(embeddings_dir):
        return []
    return sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(embeddings_dir)
        if name.endswith(".npy")
    )


@server.get("/")
async def get_status():
    STATUS.rvc = list_rvc_models()
    STATUS.uvr = list_uvr_models()
    STATUS.denoise = list_uvr_denoise_models()
    STATUS.tts = TTS_METHODS
    STATUS.speakers = _list_tts_speakers()
    gc_collect()
    return STATUS


@server.get("/rvc")
async def get_rvc():
    gc_collect()
    return list_rvc_models()


@server.post("/rvc/{name}")
async def rvc_infer(body: RVCInferenceParams, name: str):
    response = {}
    gc_collect()
    output_audio = convert_vocals(name=name, **_model_data(body))
    if output_audio:
        response["data"] = audio2bytes(*output_audio)
    gc_collect()
    return response


@server.get("/uvr")
async def get_uvr():
    gc_collect()
    return list_uvr_models()


@server.get("/uvr/preprocess")
async def get_uvr_preprocess():
    gc_collect()
    return list_uvr_denoise_models()


@server.get("/uvr/postprocess")
async def get_uvr_postprocess():
    gc_collect()
    return list_uvr_denoise_models()


@server.post("/uvr")
async def uvr_infer(body: UVRInferenceParams):
    response = {}
    gc_collect()
    result = split_vocals(**_model_data(body))
    if result:
        vocals, instrumentals = result
        response["vocals"] = audio2bytes(*vocals)
        response["instrumentals"] = audio2bytes(*instrumentals)
    gc_collect()
    return response


@server.get("/tts")
async def get_tts():
    gc_collect()
    return {"methods": TTS_METHODS, "speakers": _list_tts_speakers()}


@server.post("/tts")
async def tts_infer(body: TTSInferenceParams):
    response = {}
    gc_collect()
    output_audio = generate_speech(**_model_data(body))
    if output_audio:
        response["data"] = audio2bytes(*output_audio)
    gc_collect()
    return response


@server.get("/jobs")
async def get_jobs():
    return list_jobs()


@server.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@server.get("/jobs/{job_id}/logs")
async def get_job_output(job_id: str, tail: int | None = 200):
    logs = get_job_logs(job_id, tail=tail)
    if logs is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return logs


@server.post("/jobs/preprocess")
async def start_preprocess_job(body: PreprocessJobParams):
    return submit_preprocess_job(_model_data(body))


@server.post("/jobs/extract")
async def start_extract_job(body: ExtractFeaturesJobParams):
    return submit_extract_job(_model_data(body))


@server.post("/jobs/train")
async def start_train_job(body: TrainJobParams):
    return submit_train_job(_model_data(body))


@server.post("/jobs/index")
async def start_index_job(body: TrainIndexJobParams):
    return submit_index_job(_model_data(body))


@server.post("/jobs/speaker-embedding")
async def start_speaker_embedding_job(body: SpeakerEmbeddingJobParams):
    return submit_speaker_embedding_job(_model_data(body))


def main():
    gc_collect()
    parser = argparse.ArgumentParser(description="Start API server to run RVC and UVR")
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=get_optimal_threads(1),
        help="Number of workers to use",
        required=False,
    )
    parser.add_argument(
        "-r",
        "--reload",
        action="store_true",
        default=False,
        help="Reloads on change",
        required=False,
    )
    parser.add_argument(
        "-p", "--port", type=int, default=5555, help="Port of server", required=False
    )
    parser.add_argument(
        "-d",
        "--host",
        type=str,
        default="localhost",
        help="Domain of server",
        required=False,
    )
    args = parser.parse_args()

    # cmd=f"{config.python_cmd} -m uvicorn api:server {'--reload' if args.reload else ''} --workers={args.workers} --port={args.port} --host={args.host}"
    cmd = [
        config.python_cmd,
        "-m",
        "uvicorn",
        "services.ml.api:server",
        f"--workers={args.workers}",
        f"--port={args.port}",
        f"--host={args.host}",
    ]
    if args.reload:
        cmd += ["--reload"]
    subprocess.call(cmd, cwd=BASE_DIR)


if __name__ == "__main__":
    main()
