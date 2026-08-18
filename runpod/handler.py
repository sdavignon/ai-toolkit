import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import runpod
import yaml

ROOT = Path(os.getenv("AITK_ROOT", "/app/ai-toolkit"))
VOLUME = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
TRAINING_ROOT = VOLUME / "lora-training"
LORA_ROOT = VOLUME / "loras"


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    if not value:
        raise ValueError("name must contain at least one letter or number")
    return value[:96]


def _download(url: str, target: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "ai-toolkit-runpod/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response, open(target, "wb") as f:
        shutil.copyfileobj(response, f)


def _prepare_dataset(job_input: dict, dataset_dir: Path, trigger: str) -> int:
    images = job_input.get("images") or []
    if not images:
        raise ValueError("input.images must contain at least one training image")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for index, item in enumerate(images, start=1):
        if isinstance(item, str):
            url = item
            caption = trigger
        elif isinstance(item, dict):
            url = item.get("url")
            caption = item.get("caption") or trigger
        else:
            raise ValueError(f"images[{index - 1}] must be a URL string or object")

        if not url or not str(url).startswith(("http://", "https://")):
            raise ValueError(f"images[{index - 1}] is missing a valid http(s) url")

        suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            suffix = ".jpg"

        stem = f"{index:04d}"
        image_path = dataset_dir / f"{stem}{suffix}"
        caption_path = dataset_dir / f"{stem}.txt"
        _download(url, image_path)
        caption_path.write_text(str(caption).strip(), encoding="utf-8")
        count += 1

    return count


def _make_config(job_input: dict, name: str, trigger: str, dataset_dir: Path, output_dir: Path) -> dict:
    steps = int(job_input.get("steps", 1800))
    rank = int(job_input.get("rank", 16))
    save_every = int(job_input.get("save_every", 300))
    sample_every = int(job_input.get("sample_every", 300))
    lr = float(job_input.get("lr", 1e-4))
    model = job_input.get("model", "black-forest-labs/FLUX.1-dev")
    sample_prompts = job_input.get("sample_prompts") or [
        "[trigger] portrait, neutral background, natural light",
        "[trigger] full body, outdoors, documentary photography",
        "[trigger] in a laboratory, educational photography",
    ]

    return {
        "job": "extension",
        "config": {
            "name": name,
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": str(output_dir),
                    "device": "cuda:0",
                    "trigger_word": trigger,
                    "network": {"type": "lora", "linear": rank, "linear_alpha": rank},
                    "save": {
                        "dtype": "float16",
                        "save_every": save_every,
                        "max_step_saves_to_keep": int(job_input.get("max_checkpoints", 4)),
                        "push_to_hub": False,
                    },
                    "datasets": [
                        {
                            "folder_path": str(dataset_dir),
                            "caption_ext": "txt",
                            "caption_dropout_rate": float(job_input.get("caption_dropout_rate", 0.05)),
                            "shuffle_tokens": False,
                            "cache_latents_to_disk": True,
                            "resolution": job_input.get("resolution", [512, 768, 1024]),
                        }
                    ],
                    "train": {
                        "batch_size": int(job_input.get("batch_size", 1)),
                        "steps": steps,
                        "gradient_accumulation_steps": int(job_input.get("gradient_accumulation_steps", 1)),
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": job_input.get("optimizer", "adamw8bit"),
                        "lr": lr,
                        "ema_config": {"use_ema": True, "ema_decay": 0.99},
                        "dtype": job_input.get("dtype", "bf16"),
                    },
                    "model": {
                        "name_or_path": model,
                        "is_flux": True,
                        "quantize": bool(job_input.get("quantize", True)),
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": sample_every,
                        "sample_start_step": 0,
                        "width": int(job_input.get("sample_width", 1024)),
                        "height": int(job_input.get("sample_height", 1024)),
                        "prompts": sample_prompts,
                        "neg": "",
                        "seed": int(job_input.get("seed", 42)),
                        "walk_seed": True,
                        "guidance_scale": float(job_input.get("guidance_scale", 4)),
                        "sample_steps": int(job_input.get("sample_steps", 20)),
                    },
                }
            ],
        },
        "meta": {"name": "[name]", "version": "1.0", "runpod": True},
    }


def _find_weights(output_dir: Path) -> list[Path]:
    weights = list(output_dir.rglob("*.safetensors"))
    return sorted(weights, key=lambda p: p.stat().st_mtime)


def handler(job):
    job_input = job.get("input") or {}
    name = _slug(str(job_input.get("name", "flux-lora")))
    trigger = str(job_input.get("trigger_word") or name).strip()
    lora_type = _slug(str(job_input.get("type", "character")))

    work_dir = TRAINING_ROOT / name
    dataset_dir = work_dir / "dataset"
    output_dir = work_dir / "output"
    config_dir = work_dir / "config"

    if job_input.get("clean", False) and work_dir.exists():
        shutil.rmtree(work_dir)

    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_count = _prepare_dataset(job_input, dataset_dir, trigger)
    config = _make_config(job_input, name, trigger, dataset_dir, output_dir)
    config_path = config_dir / "train.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    command = ["python", str(ROOT / "run.py"), str(config_path)]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_path = work_dir / "training.log"
    log_path.write_text(completed.stdout or "", encoding="utf-8")

    if completed.returncode != 0:
        tail = "\n".join((completed.stdout or "").splitlines()[-80:])
        return {
            "status": "failed",
            "name": name,
            "error": "AI Toolkit training failed",
            "return_code": completed.returncode,
            "log_path": str(log_path),
            "log_tail": tail,
        }

    weights = _find_weights(output_dir)
    if not weights:
        return {
            "status": "failed",
            "name": name,
            "error": "Training completed but no .safetensors file was found",
            "log_path": str(log_path),
        }

    destination_dir = LORA_ROOT / lora_type / name
    destination_dir.mkdir(parents=True, exist_ok=True)
    published = []
    for weight in weights:
        destination = destination_dir / weight.name
        shutil.copy2(weight, destination)
        published.append(str(destination))

    manifest = {
        "status": "completed",
        "name": name,
        "type": lora_type,
        "trigger_word": trigger,
        "image_count": image_count,
        "model": job_input.get("model", "black-forest-labs/FLUX.1-dev"),
        "lora_path": published[-1],
        "checkpoints": published,
        "training_dir": str(work_dir),
        "log_path": str(log_path),
    }
    (destination_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


runpod.serverless.start({"handler": handler})
