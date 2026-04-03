from typing import List, Literal
from pydantic import BaseModel


class RVCInferenceParams(BaseModel):
    audio_data: str
    f0_up_key: int = 0
    f0_method: List[Literal["rmvpe", "rmvpe+", "crepe", "mangio-crepe"]] = ["rmvpe"]
    f0_autotune: bool = False
    merge_type: Literal["median", "mean", "min", "max"] = "median"
    index_rate: float = 0.75
    resample_sr: int = 0
    rms_mix_rate: float = 0.25
    protect: float = 0.25
    filter_radius: int = 3


class UVRInferenceParams(BaseModel):
    uvr_models: List[str]
    audio_data: str
    preprocess_models: List[str] = []
    postprocess_models: List[str] = []
    agg: int = 10
    merge_type: Literal["median", "mean", "min", "max"] = "median"
    use_cache: bool = True
    format: Literal["mp3", "flac", "wav"] = "flac"


class UVRRVCInferenceParams(BaseModel):
    uvr_params: UVRInferenceParams
    rvc_params: RVCInferenceParams
    audio_data: str


class TTSInferenceParams(BaseModel):
    text: str
    speaker: str | None = None
    method: Literal["speecht5", "bark", "tacotron2", "edge", "vits", "silero"] = (
        "speecht5"
    )
    device: Literal["cpu", "cuda"] = "cpu"
    dialog_only: bool = False


class PreprocessJobParams(BaseModel):
    exp_dir: str
    sr: Literal["32k", "40k", "48k"] = "40k"
    trainset_dir: str
    n_threads: int = 1
    version: Literal["v1", "v2"] = "v2"
    period: float = 3.0
    overlap: float = 0.3


class ExtractFeaturesJobParams(BaseModel):
    exp_dir: str
    n_threads: int = 1
    version: Literal["v1", "v2"] = "v2"
    if_f0: bool = True
    f0method: List[Literal["rmvpe", "rmvpe+", "crepe", "mangio-crepe"]] = ["rmvpe"]
    device: Literal["cpu", "cuda"] = "cpu"
    sr: Literal["32k", "40k", "48k"] = "40k"


class TrainJobParams(BaseModel):
    exp_dir: str
    if_f0: bool = True
    spk_id: int = 0
    version: Literal["v1", "v2"] = "v2"
    sr: Literal["32k", "40k", "48k"] = "40k"
    gpus: str = ""
    batch_size: int
    total_epoch: int
    save_epoch: int
    pretrained_G: str | None = None
    pretrained_D: str | None = None
    if_save_latest: bool = True
    if_cache_gpu: bool = False
    if_save_every_weights: bool = True


class TrainIndexJobParams(BaseModel):
    exp_dir: str
    version: Literal["v1", "v2"] = "v2"
    sr: Literal["32k", "40k", "48k"] = "40k"


class SpeakerEmbeddingJobParams(BaseModel):
    exp_dir: str
    version: Literal["v1", "v2"] = "v2"
    sr: Literal["32k", "40k", "48k"] = "40k"
