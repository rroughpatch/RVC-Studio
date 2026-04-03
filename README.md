> :warning: **The main branch is bleeding edge**: Expect frequent updates and many breaking changes after every commit

# RVC Studio
This project provides a comprehensive platform for training RVC models and generating AI voice covers. Use the app to download the required files before using or manually download them here: https://huggingface.co/datasets/SayanoAI/RVC-Studio/tree/main

## Monorepo Migration
This repository is being restructured into a browser-first monorepo:

- `apps/web`: Vite + React frontend
- `services/gateway`: Bun + Elysia local API and orchestration layer
- `services/ml`: Python compute service for RVC, UVR, TTS, and training
- `packages/shared`: shared TypeScript types and constants

Current status:

- The workspace now uses Vite+ (`vp`) for root task orchestration and staged-file hooks.
- Package-level runtime commands can still use Bun where that makes sense.
- The existing Python runtime still lives at the repo root for compatibility.
- `models`, `songs`, `output`, `logs`, `datasets`, and `configs` remain top-level data directories during migration.

## Features
* Youtube music downloader: download any music video from Youtube as an mp3 file with just one click.
* 1-click AI song covers: easily create AI song covers using RVC.
* RVC Model fine-tuning: fine-tune an RVC model to mimic any voice you want using your own data.
* 1-click TTS using RVC model: convert any text to speech using the fine-tuned VC model with just one click.
* Built-in tensorboard: You can monitor the training progress and performance of your VC model using a built-in tensorboard dashboard.
* LLM integration: chat with your RVC model in real time using popular LLMs.
* Auto-Playlist: let your RVC model sing songs from your favourite playlist.

## Planned Features
* Demucs: Meta's vocals and instrumental music source separation.
* Audio-postprocessing: You can enhance the quality of your generated songs by adding reverbs, echos, etc.
* TTS using cloud API: use a cloud-based text-to-speech service to generate high-quality and natural-sounding speech from any text.
* Real-time VC interface: convert your voice using your favourite RVC model.

## Requirements
- Python 3.10
- [uv](https://docs.astral.sh/uv/)
- System packages for audio and TTS support:
  `ffmpeg`, `espeak`, `libportaudio2`, `portaudio19-dev`, `python3-dev`

## Easy Install
1. Clone this repository or download the zip file and extract it.
2. Double-click `uv-install.bat` on Windows to install `uv` if needed, create the local `.venv`, and launch the app.

## Manual Installation
1. Clone this repository or download the zip file.
2. Install `uv` if it is not already available: https://docs.astral.sh/uv/getting-started/installation/
3. Navigate to the project directory and install Python 3.10 with `uv python install 3.10`.
4. Sync the project environment with `uv sync`.
5. Run the Streamlit app with `uv run streamlit run Home.py`.

To start the ML API service directly, use `uv run python -m services.ml`.

Or run it in [Google Colab](https://colab.research.google.com/github/SayanoAI/RVC-Studio/blob/master/RVC_Studio.ipynb)

## Instructions for Inference page
1. Download all the required models on the webui page or here: https://huggingface.co/datasets/SayanoAI/RVC-Studio/tree/main
2. Put your favourite songs in the ./songs folder
3. Navigate to "RVC Server" page and start the ML service
4. Navigate to "Inference" page and press "Refresh Data" button
5. Select a song (only wav/flac/ogg/mp3 are supported for now)
6. Select a voice model (put your RVC v2 models in ./models/RVC/ and index file in ./models/RVC/.index/)
7. Choose a vocal extraction model (preprocessing model is optional)
8. Click "Save Options" and "1-Click VC" to get started

## Instructions for Chat page
Chat functionality has been migrated to [RVC-Chat](https://github.com/SayanoAI/RVC-Chat).

**Feel free to use larger versions of these models if your computer can handle it. (you will have to build your own config)**

## Dockerize
Run `docker compose up --build` in the main project folder.

~~**Known issue:** Tensorboard doesn't work inside a docker container. Feel free to submit a PR if you know a solution.~~ fixed in commit 8b720936b4dab347cba0e4a791330fb533bfdf1d 

## FAQs
* Trouble with ffmpeg/espeak? [Read this](/dist/README.md)

## Disclaimer
This project is for educational and research purposes only. The generated voice overs are not intended to infringe on any copyrights or trademarks of the original songs or text. The project does not endorse or promote any illegal or unethical use of the generative AI technology. The project is not responsible for any damages or liabilities arising from the use or misuse of the generated voice overs.

## Credits
This project uses code and AI models from the following repositories:
- [Karafan](https://github.com/Captain-FLAM/KaraFan) by Captain-FLAM.
- [Retrieval-based Voice Conversion WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) by RVC-Project.
- [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui) by Anjok07.
- [Streamlit](https://github.com/streamlit/streamlit) by streamlit.
- [Tacotron 2 - PyTorch implementation with faster-than-realtime inference](https://github.com/NVIDIA/tacotron2) by NVIDIA. 
- [Bark: A Speech Synthesis Toolkit for Bengali Language](https://github.com/suno-ai/bark) by suno-ai.
- [SpeechT5: A Self-Supervised Pre-training Model for Speech Recognition and Generation](https://github.com/microsoft/SpeechT5) by microsoft.
- [VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://github.com/jaywalnut310/vits) by jaywalnut310 et al.
- [Applio-RVC-Fork](https://github.com/IAHispano/Applio-RVC-Fork) by IAHispano

We thank all the authors and contributors of these repositories for their amazing work and for making their code and models publicly available.
