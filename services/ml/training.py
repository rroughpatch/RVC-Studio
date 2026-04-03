import datetime
import os
import shlex
import subprocess
from random import shuffle

import faiss
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

try:
    from .bootstrap import ensure_repo_root
except ImportError:
    from services.ml.bootstrap import ensure_repo_root

ensure_repo_root()

from lib import BASE_DIR, BASE_MODELS_DIR, LOG_DIR, config, i18n
from lib.audio import load_input_audio, save_input_audio
from .preprocessing_utils import extract_features_trainset, preprocess_trainset
from .tts_cli import EMBEDDING_CHECKPOINT, TTS_MODELS_DIR

SR_MAP = {"32k": 32000, "40k": 40000, "48k": 48000}


def get_model_log_name(exp_dir: str, version: str, sr: str) -> str:
    return f"{exp_dir}_{version}_{sr}"


def get_model_log_dir(exp_dir: str, version: str, sr: str) -> str:
    return os.path.join(LOG_DIR, get_model_log_name(exp_dir, version, sr))


def preprocess_data(
    exp_dir: str,
    sr: str,
    trainset_dir: str,
    n_threads: int,
    version: str,
    period: float = 3.0,
    overlap: float = 0.3,
):
    model_log_dir = get_model_log_dir(exp_dir, version, sr)
    os.makedirs(model_log_dir, exist_ok=True)
    return preprocess_trainset(
        trainset_dir, SR_MAP[sr], n_threads, model_log_dir, period, overlap
    )


def extract_features(
    exp_dir: str,
    n_threads: int,
    version: str,
    if_f0: bool,
    f0method,
    device: str,
    sr: str,
):
    model_log_dir = get_model_log_dir(exp_dir, version, sr)
    os.makedirs(model_log_dir, exist_ok=True)

    n_p = max(
        n_threads // (len(f0method) if isinstance(f0method, list) else os.cpu_count()),
        1,
    )

    if isinstance(f0method, list):
        return "\n".join(
            [
                extract_features_trainset(
                    model_log_dir,
                    n_p=n_p,
                    f0method=method,
                    device=device,
                    if_f0=if_f0,
                    version=version,
                )
                for method in f0method
            ]
        )
    return extract_features_trainset(
        model_log_dir,
        n_p=n_p,
        f0method=f0method,
        device=device,
        if_f0=if_f0,
        version=version,
    )


def create_filelist(exp_dir: str, if_f0: bool, spk_id: int, version: str, sr: str):
    model_log_dir = get_model_log_dir(exp_dir, version, sr)

    print(i18n("training.create_filelist"))
    gt_wavs_dir = os.sep.join([model_log_dir, "0_gt_wavs"])
    feature_dir = os.sep.join(
        [model_log_dir, "3_feature256" if version == "v1" else "3_feature768"]
    )
    os.makedirs(gt_wavs_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    if if_f0:
        f0_dir = os.sep.join([model_log_dir, "2a_f0"])
        f0nsf_dir = os.sep.join([model_log_dir, "2b-f0nsf"])
        names = (
            {os.path.splitext(name)[0] for name in os.listdir(feature_dir)}
            & {os.path.splitext(name)[0] for name in os.listdir(f0_dir)}
            & {os.path.splitext(name)[0] for name in os.listdir(f0nsf_dir)}
        )
    else:
        names = {os.path.splitext(name)[0] for name in os.listdir(feature_dir)}
    opt = []
    missing_data = []
    for name in names:
        name_parts = name.split(",")
        gt_name = name if len(name_parts) == 1 else name_parts[-1]
        gt_file = os.path.join(gt_wavs_dir, gt_name)
        if not os.path.isfile(gt_file):
            print(f"{gt_name} not found!")
            missing_data.append(gt_name)
            continue

        if if_f0:
            data = "|".join(
                [
                    gt_file,
                    os.path.join(feature_dir, f"{name}.npy"),
                    os.path.join(f0_dir, f"{name}.npy"),
                    os.path.join(f0nsf_dir, f"{name}.npy"),
                    str(spk_id),
                ]
            )
        else:
            data = "|".join(
                [gt_file, os.path.join(feature_dir, f"{name}.npy"), str(spk_id)]
            )
        opt.append(data)

    fea_dim = 256 if version == "v1" else 768
    num_mute = max(2, len(opt) // 100)
    for _ in range(num_mute):
        if if_f0:
            data = "|".join(
                [
                    os.path.join(LOG_DIR, "mute", "0_gt_wavs", f"mute{sr}.wav"),
                    os.path.join(LOG_DIR, "mute", f"3_feature{fea_dim}", "mute.npy"),
                    os.path.join(LOG_DIR, "mute", "2a_f0", "mute.wav.npy"),
                    os.path.join(LOG_DIR, "mute", "2b-f0nsf", "mute.wav.npy"),
                    str(spk_id),
                ]
            )
        else:
            data = "|".join(
                [
                    os.path.join(LOG_DIR, "mute", "0_gt_wavs", f"mute{sr}.wav"),
                    os.path.join(LOG_DIR, "mute", f"3_feature{fea_dim}", "mute.npy"),
                    str(spk_id),
                ]
            )
        opt.append(data)

    shuffle(opt)
    if missing_data:
        raise RuntimeError(
            f"missing ground truth data: {len(opt)=}, {len(missing_data)=}"
        )

    with open(os.path.join(model_log_dir, "filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(opt))
    print("write filelist done")
    return True


def build_training_command(
    exp_dir: str,
    if_f0: bool,
    spk_id: int,
    version: str,
    sr: str,
    gpus: str,
    batch_size: int,
    total_epoch: int,
    save_epoch: int,
    pretrained_G: str | None,
    pretrained_D: str | None,
    if_save_latest: bool,
    if_cache_gpu: bool,
    if_save_every_weights: bool,
):
    print(i18n("training.train_model"))
    create_filelist(exp_dir, if_f0, spk_id, version, sr)

    cmd = [
        config.python_cmd,
        "-m",
        "services.ml.training_cli",
        "-e",
        get_model_log_name(exp_dir, version, sr),
        "-n",
        exp_dir,
        "-sr",
        sr,
        "-f0",
        str(1 if if_f0 else 0),
        "-bs",
        str(batch_size),
        "-te",
        str(total_epoch),
        "-se",
        str(save_epoch),
        "-l",
        str(1 if if_save_latest else 0),
        "-c",
        str(1 if if_cache_gpu else 0),
        "-sw",
        str(1 if if_save_every_weights else 0),
        "-v",
        version,
    ]
    if gpus:
        cmd.extend(["-g", gpus])
    if pretrained_G and gpus:
        cmd.extend(["-pg", pretrained_G])
    if pretrained_D and gpus:
        cmd.extend(["-pd", pretrained_D])
    return cmd


def get_training_log_path(exp_dir: str, version: str, sr: str) -> str:
    return os.path.join(get_model_log_dir(exp_dir, version, sr), "train.log")


def start_training_process(**kwargs):
    cmd = build_training_command(**kwargs)
    train_log = get_training_log_path(
        kwargs["exp_dir"], kwargs["version"], kwargs["sr"]
    )

    with open(train_log, "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    return process, cmd, train_log


def train_model(**kwargs):
    try:
        _, cmd, train_log = start_training_process(**kwargs)
        command_str = " ".join(shlex.quote(part) for part in cmd)
        return (
            f"Successfully started training with {command_str}. "
            f"Logs are being written to {train_log}."
        )
    except Exception as exc:
        return f"Failed to initiate training: {exc}"


def train_index(exp_dir: str, version: str, sr: str):
    try:
        model_log_dir = get_model_log_dir(exp_dir, version, sr)
        feature_dir = os.sep.join(
            [model_log_dir, "3_feature256" if version == "v1" else "3_feature768"]
        )
        os.makedirs(feature_dir, exist_ok=True)

        npys = []
        for name in sorted(os.listdir(feature_dir)):
            phone = np.load(os.sep.join([feature_dir, name]))
            npys.append(phone)
        big_npy = np.concatenate(npys, 0)

        big_npy_idx = np.arange(big_npy.shape[0])
        np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]

        if big_npy.shape[0] > 2e5:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )

        np.save(os.path.join(model_log_dir, "total_fea.npy"), big_npy)

        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
        print(f"{big_npy.shape},{n_ivf}")
        index = faiss.index_factory(
            256 if version == "v1" else 768, f"IVF{n_ivf},Flat"
        )
        print("training index")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        index.train(big_npy)
        faiss.write_index(
            index,
            os.path.join(
                model_log_dir,
                f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir}_{version}.index",
            ),
        )
        print("adding index")
        batch_size_add = 8192
        for i in range(0, big_npy.shape[0], batch_size_add):
            index.add(big_npy[i : i + batch_size_add])

        index_name = os.path.join(
            BASE_MODELS_DIR, "RVC", ".index", f"{exp_dir}_{version}_{sr}.index"
        )
        faiss.write_index(index, index_name)

        return f"saved index file to {index_name}"
    except Exception as exc:
        return f"Failed to train index: {exc}"


def train_speaker_embedding(exp_dir: str, version: str, sr: str):
    try:
        model_log_dir = get_model_log_name(exp_dir, version, sr)
        training_file = os.path.join(LOG_DIR, model_log_dir, "embedding.wav")
        if os.path.isfile(training_file):
            audio = load_input_audio(training_file, sr=16000, mono=True)[0]
        else:
            dataset_dir = os.path.join(LOG_DIR, model_log_dir, "1_16k_wavs")
            audio = np.concatenate(
                [
                    load_input_audio(
                        os.path.join(dataset_dir, fname), sr=16000, mono=True
                    )[0]
                    for fname in os.listdir(dataset_dir)
                ],
                axis=None,
            )
            save_input_audio(training_file, (audio, 16000))

        from speechbrain.pretrained import EncoderClassifier

        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(
            TTS_MODELS_DIR, EMBEDDING_CHECKPOINT
        )
        classifier = EncoderClassifier.from_hparams(
            source=EMBEDDING_CHECKPOINT,
            savedir=os.path.join(TTS_MODELS_DIR, EMBEDDING_CHECKPOINT),
        )
        embeddings = classifier.encode_batch(
            torch.from_numpy(audio), normalize=True
        ).squeeze(0)

        embedding_path = os.path.join(TTS_MODELS_DIR, "embeddings", f"{exp_dir}.npy")
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        np.save(embedding_path, embeddings.numpy())
        return f"Saved speaker embedding to: {embedding_path}"
    except Exception as exc:
        return f"Failed to train speecht5 speaker embedding: {exc}"


def write_job_header(log_path: str, label: str, metadata: dict | None = None):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        started_at = datetime.datetime.now().isoformat(timespec="seconds")
        handle.write(f"[{started_at}] {label}\n")
        if metadata:
            for key, value in metadata.items():
                handle.write(f"{key}={value}\n")
