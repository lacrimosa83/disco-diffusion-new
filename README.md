# Disco Diffusion - Optimized Version

An optimized and refactored version of Disco Diffusion for AI image generation.

## 📁 Project Structure

```
Disco Difussion NEW/
├── config.py           # Configuration settings
├── main.py             # Main entry point (verify setup)
├── generate.py         # Image generation script
├── requirements.txt    # Python dependencies
├── README.md          # This file
│
├── CLIP/              # CLIP model code
├── SLIP/              # SLIP model code
├── ResizeRight/       # Image resizing utilities
├── guided-diffusion/  # Diffusion model code
│
└── content/           # Model weights and outputs
    ├── 512x512_diffusion_uncond_finetune_008100.pt  # Main diffusion model (~2GB)
    ├── secondary_model_imagenet_2.pth               # Secondary model (~50MB)
    └── output/           # Generated images go here
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Copy Model Files

Copy these files from the original `Disco Difussion/content/` folder:
- `512x512_diffusion_uncond_finetune_008100.pt` (~2.2 GB)
- `secondary_model_imagenet_2.pth` (~55 MB)

Or download from:
- https://models.rlab.be/

### 3. Configure

Edit `config.py` to set your prompts and settings:

```python
text_prompts = {
    "A beautiful landscape at sunset": 1.0,
    "detailed, 8k, unreal engine": 0.5,
}
```

### 4. Run

```bash
python main.py
```

## ⚠️ Important Notes

1. **GPU Required** - This needs a GPU with 8GB+ VRAM
2. **Memory Management** - Set `max_gpu_memory` in config.py to prevent blue screen
3. **Model Files** - You must copy the model files manually (too large to include)

## 🔧 Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| `width` | Image width (multiple of 64) | 1280 |
| `height` | Image height (multiple of 64) | 768 |
| `iterations` | Diffusion steps | 500 |
| `guidance_scale` | Prompt strength | 7.5 |
| `clip_guidance_scale` | CLIP guidance | 5000 |
| `max_gpu_memory` | GPU memory limit (0.0-1.0) | 0.8 |

## 🛠️ Troubleshooting

### Blue Screen / OOM Errors

- Reduce `max_gpu_memory` to 0.5 or 0.6
- Reduce image size
- Reduce `cutn` parameter

### Import Errors

Make sure all subdirectories (CLIP, guided-diffusion, etc.) are present.

### Model Not Found

Copy the `.pt` files from the original project's `content/` folder.

## 📝 Original Project

This is an optimized version of the original Disco Diffusion:
- Original: https://github.com/alembics/disco-diffusion
