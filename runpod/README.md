# RunPod Serverless FLUX LoRA Trainer

This folder adds a RunPod Serverless entrypoint for AI Toolkit FLUX LoRA training.

## Build

Use the repository root as the Docker build context:

```bash
docker build -f runpod/Dockerfile -t sdavignon/ai-toolkit-runpod:latest .
```

For RunPod Hub, set the Dockerfile path to:

```text
runpod/Dockerfile
```

## Required RunPod configuration

- Attach a Network Volume mounted at `/runpod-volume`.
- Add `HF_TOKEN` as a RunPod Secret/environment variable. FLUX.1-dev is gated on Hugging Face and the account must have accepted the model license.
- Use a GPU with at least 24 GB VRAM for the default FLUX configuration.
- The image uses CUDA 13.0 / PyTorch cu130 and therefore requires a compatible NVIDIA driver on the worker.

The Hugging Face cache is persisted under `/runpod-volume/huggingface` so subsequent workers can reuse downloaded model files.

## API input

```json
{
  "input": {
    "name": "sam-baby-biologist",
    "type": "character",
    "trigger_word": "SAMBB",
    "images": [
      {
        "url": "https://example.com/sam-01.jpg",
        "caption": "SAMBB standing in a biology laboratory, medium shot"
      },
      {
        "url": "https://example.com/sam-02.jpg",
        "caption": "SAMBB outdoors holding a field notebook"
      }
    ],
    "steps": 1800,
    "rank": 16,
    "save_every": 300,
    "sample_every": 300,
    "sample_prompts": [
      "[trigger] in a biology laboratory, documentary photography",
      "[trigger] outdoors examining an insect, full body"
    ]
  }
}
```

`images` may also be a list of URL strings. When no caption is supplied, the trigger word is used as the caption.

Supported image URL suffixes are jpg, jpeg, and png. If a URL does not expose one of those suffixes, the downloaded payload is saved as jpg, so image delivery URLs should preferably end with their real image extension.

## Output

Successful jobs return a manifest similar to:

```json
{
  "status": "completed",
  "name": "sam-baby-biologist",
  "type": "character",
  "trigger_word": "SAMBB",
  "image_count": 24,
  "model": "black-forest-labs/FLUX.1-dev",
  "lora_path": "/runpod-volume/loras/character/sam-baby-biologist/sam-baby-biologist.safetensors",
  "checkpoints": [
    "/runpod-volume/loras/character/sam-baby-biologist/sam-baby-biologist_000000300.safetensors"
  ],
  "training_dir": "/runpod-volume/lora-training/sam-baby-biologist",
  "log_path": "/runpod-volume/lora-training/sam-baby-biologist/training.log"
}
```

The exact checkpoint filenames are determined by AI Toolkit.

## Shared volume with inference

The intended production setup is for the LoRA trainer and the FLUX Krea inference endpoint to use the same RunPod Network Volume. A completed training job writes models under:

```text
/runpod-volume/loras/<type>/<name>/
```

The inference endpoint can then use the returned `lora_path` directly.

## Optional inputs

- `model`: defaults to `black-forest-labs/FLUX.1-dev`
- `steps`: default `1800`
- `rank`: default `16`
- `lr`: default `1e-4`
- `batch_size`: default `1`
- `gradient_accumulation_steps`: default `1`
- `save_every`: default `300`
- `sample_every`: default `300`
- `max_checkpoints`: default `4`
- `resolution`: default `[512, 768, 1024]`
- `quantize`: default `true`
- `clean`: remove an existing training folder for the same name before starting

Do not send Hugging Face tokens in job input. Store `HF_TOKEN` as a RunPod Secret.
