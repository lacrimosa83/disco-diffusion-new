# Disco Diffusion - Optimized Configuration
# AI Image Generation from Text Prompts

# ==================== Image Settings ====================
image_settings = {
    "width": 1280,          # Image width (multiple of 64)
    "height": 768,          # Image height (multiple of 64) 
    "init_image": None,     # Path to init image (or None for random)
    "init_scale": 1000,     # Init image prompt strength
    "skip_steps": 0,        # Skip first N steps (for resume)
}

# ==================== Prompt Settings ====================
text_prompts = {
    "A beautiful landscape of mountains at sunset, trending on artstation": 1.0,
    "detailed, 8k, unreal engine 5": 0.5,
}

# ==================== Model Settings ====================
model_settings = {
    "diffusion_model": "512x512_diffusion_uncond_finetune_008100",
    "use_secondary_model": True,
    "timestep_respacing": "ddim250",  # ddim25, ddim50, ddim100, ddim250, ddim500, ddim1000
    "diffusion_steps": 1000,
    "use_checkpoint": False,
}

# ==================== CLIP Settings ====================
clip_settings = {
    "ViTB32": True,
    "ViTB16": True,
    "RN101": False,
    "RN50": False,
    "RN50x4": False,
    "RN50x16": False,
}

# ==================== Generation Settings ====================
generation_settings = {
    "batch_size": 1,           # Number of images to generate
    "num_batches": 1,          # Number of batches
    "iterations": 500,         # Steps per image
    "guidance_scale": 7.5,     # Prompt strength (higher = more literal)
    "clip_guidance_scale": 5000,# CLIP guidance strength
    "tv_scale": 150,           # Total variation loss
    "range_scale": 150,        # Color range loss
    "sat_scale": 0,            # Saturation loss
    "cutn": 16,               # Number of cuts
    "cut_pow": 1,             # Cut power
}

# ==================== Perlin Noise Settings ====================
perlin_settings = {
    "perlin_init": False,      # Start with perlin noise
    "perlin_mode": "color",   # color, gray, or mixed
}

# ==================== Output Settings ====================
output_settings = {
    "output_folder": "./output",
    "batch_name": "disco",
    "save_frequency": 50,       # Save every N steps
    "generate_gif": False,     # Generate animated GIF
    "display_frequency": 100,  # Display progress every N steps
}

# ==================== Performance Settings ====================
performance_settings = {
    "seed": None,              # Random seed (None for random)
    "use_amp": True,           # Automatic Mixed Precision
    "device": "cuda",         # cuda or cpu
    "max_gpu_memory": 0.8,    # Max GPU memory usage (0.0-1.0)
    "clear_memory_every": 50, # Clear cache every N steps
}

# ==================== Advanced Settings ====================
advanced_settings = {
    "fuzzy_prompt": False,     # Add noise to prompts
    "rand_mag": 0.1,          # Random magnitude
    "skip_augs": False,        # Skip augmentations
    "randomize_class": True,
    "clip_denoised": False,
    "rand_mag": 0.1,
}
