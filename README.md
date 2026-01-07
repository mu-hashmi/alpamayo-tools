# alpamayo-tools

Unofficial community tools for NVIDIA's [Alpamayo-R1](https://developer.nvidia.com/drive/alpamayo) and [PhysicalAI-AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) ecosystem.

## Features

- **PyTorch Dataset** - Clean `torch.utils.data.Dataset` wrapper for PhysicalAI-AV
- **Alpamayo Inference** - Simple API for running Alpamayo-R1 inference
- **CoC Embeddings** - Chain-of-Cognition text embedding pipeline

## Installation

```bash
pip install alpamayo-tools
```

For inference capabilities (requires ~24GB+ GPU):
```bash
pip install alpamayo-tools[inference]

# Also install alpamayo_r1 from GitHub:
pip install git+https://github.com/NVlabs/alpamayo.git
```

For CoC embeddings:
```bash
pip install alpamayo-tools[embeddings]
```

## Quick Start

### PyTorch DataLoader

```python
from alpamayo_tools import PhysicalAIDataset, DatasetConfig
from torch.utils.data import DataLoader

# Configure dataset
config = DatasetConfig(
    clip_ids=["clip_001", "clip_002", "clip_003"],
    cameras=("camera_front_wide_120fov", "camera_front_tele_30fov"),
    num_frames=4,
    stream=True,  # Stream from HuggingFace
)

# Create dataset and dataloader
dataset = PhysicalAIDataset(config)
loader = DataLoader(dataset, batch_size=4, num_workers=2)

for batch in loader:
    frames = batch["frames"]  # (B, N_cameras, num_frames, 3, H, W)
    ego_history = batch["ego_history_xyz"]  # (B, 16, 3)
    # ... your training code
```

### Alpamayo Inference

```python
from alpamayo_tools.inference import AlpamayoPredictor

# Load model (requires alpamayo_r1 package)
predictor = AlpamayoPredictor.from_pretrained(
    model_id="nvidia/Alpamayo-R1-10B",
    dtype=torch.bfloat16,
)

# Run inference
result = predictor.predict_from_clip("clip_001", t0_us=5_100_000)
print(result.reasoning_text)
print(result.trajectory_xyz.shape)  # (64, 3)
```

### CoC Embeddings

```python
from alpamayo_tools import CoCEmbedder

embedder = CoCEmbedder()

texts = [
    "The vehicle ahead is braking. Reduce speed to maintain safe following distance.",
    "Clear road ahead. Continue at current speed.",
]
embeddings = embedder.embed(texts)
print(embeddings.shape)  # (2, 384)
```

### CLI: Generate Teacher Labels

```bash
# Generate labels for a list of clips
alpamayo-generate-labels \
    --clip-ids-file train_clips.parquet \
    --output-dir ./labels \
    --shard 0/2  # For multi-GPU parallelism
```

## Dataset Output Format

Each sample from `PhysicalAIDataset` contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `clip_id` | `str` | Clip identifier |
| `t0_us` | `int` | Reference timestamp (microseconds) |
| `frames` | `(N_cam, T, 3, H, W)` | Camera frames (uint8) |
| `camera_indices` | `(N_cam,)` | Camera index identifiers |
| `ego_history_xyz` | `(16, 3)` | Past trajectory in ego frame |
| `ego_history_rot` | `(16, 3, 3)` | Past rotations in ego frame |
| `ego_future_xyz` | `(64, 3)` | Future trajectory (training only) |
| `ego_future_rot` | `(64, 3, 3)` | Future rotations (training only) |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NVIDIA GPU with 24GB+ VRAM (for inference)

## Related Resources

- [Alpamayo-R1 Model](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- [PhysicalAI-AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- [Alpamayo GitHub](https://github.com/NVlabs/alpamayo)
- [AlpaSim Simulator](https://github.com/NVlabs/alpasim)

## Disclaimer

This is an unofficial community project and is not affiliated with NVIDIA Corporation.

## License

MIT License
