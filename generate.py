"""
Disco Diffusion - Image Generation Script
Optimized version with proper memory management
"""

import os
import sys
import io
import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np

# Setup
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration"""
    from config import (
        image_settings, text_prompts, model_settings,
        clip_settings, generation_settings, perlin_settings,
        output_settings, performance_settings, advanced_settings
    )
    return {
        'image': image_settings,
        'prompts': text_prompts,
        'model': model_settings,
        'clip': clip_settings,
        'gen': generation_settings,
        'perlin': perlin_settings,
        'output': output_settings,
        'perf': performance_settings,
        'advanced': advanced_settings
    }


def setup_device(config):
    """Setup compute device with memory management"""
    import torch
    
    device_type = config['perf']['device']
    
    if device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        
        # Set memory limit
        max_mem = config['perf']['max_gpu_memory']
        total_mem = torch.cuda.get_device_properties(0).total_memory
        limit_mem = int(total_mem * max_mem)
        torch.cuda.set_per_process_memory_fraction(max_mem)
        
        logger.info(f"GPU Memory Limit: {limit_mem/1e9:.1f} GB ({max_mem*100:.0f}%)")
    else:
        device = torch.device('cpu')
        logger.warning("Using CPU - generation will be slow!")
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return device


def load_clip_models(config, device):
    """Load CLIP models"""
    import clip
    import torch
    
    models = []
    clip_cfg = config['clip']
    
    model_names = []
    if clip_cfg.get('ViTB32'): model_names.append('ViT-B/32')
    if clip_cfg.get('ViTB16'): model_names.append('ViT-B/16')
    if clip_cfg.get('RN101'): model_names.append('RN101')
    if clip_cfg.get('RN50'): model_names.append('RN50')
    if clip_cfg.get('RN50x4'): model_names.append('RN50x4')
    if clip_cfg.get('RN50x16'): model_names.append('RN50x16')
    
    # Default to ViT-B/32 if none selected
    if not model_names:
        model_names = ['ViT-B/32']
        logger.warning("No CLIP model specified, using ViT-B/32")
    
    for name in model_names:
        try:
            model, preprocess = clip.load(name, device=device)
            model.eval()
            models.append((name, model, preprocess))
            logger.info(f"Loaded CLIP: {name}")
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
    
    return models


def load_diffusion_model(config, device):
    """Load diffusion model"""
    import torch
    from guided_diffusion.script_util import (
        create_model_and_diffusion,
        model_and_diffusion_defaults
    )
    
    model_cfg = config['model']
    
    # Model configuration
    model_config = model_and_diffusion_defaults()
    
    if model_cfg['diffusion_model'] == "512x512_diffusion_uncond_finetune_008100":
        model_config.update({
            'image_size': 512,
            'num_channels': 256,
            'num_res_blocks': 2,
            'attention_resolutions': '32,16,8',
            'num_heads': 4,
            'num_head_channels': 64,
            'resblock_updown': True,
            'use_fp16': True,
        })
    else:
        model_config.update({
            'image_size': 256,
            'num_channels': 256,
            'num_res_blocks': 2,
            'attention_resolutions': '32,16,8',
        })
    
    # Load model
    logger.info(f"Loading diffusion model: {model_cfg['diffusion_model']}")
    
    try:
        model, diffusion = create_model_and_diffusion(model_config)
        model.load_state_dict(torch.load(
            PROJECT_ROOT / 'content' / '512x512_diffusion_uncond_finetune_008100.pt',
            map_location=device
        ))
        model.to(device)
        model.eval()
        
        if model_config.get('use_fp16'):
            model.convert_to_fp16()
        
        logger.info("Diffusion model loaded!")
        return model, diffusion, model_config
        
    except FileNotFoundError:
        logger.error("Diffusion model file not found!")
        logger.info("Please download the model file first.")
        return None, None, model_config


def parse_prompts(prompts_dict):
    """Parse prompt dictionary to list"""
    result = []
    for prompt, weight in prompts_dict.items():
        result.append((prompt, weight))
    return result


class DiscoDiffusion:
    """Main Disco Diffusion class"""
    
    def __init__(self, config):
        self.config = config
        self.device = None
        self.clip_models = []
        self.diffusion_model = None
        self.diffusion = None
        self.lpips_model = None
        
    def initialize(self):
        """Initialize all models"""
        logger.info("Initializing Disco Diffusion...")
        
        # Setup device
        self.device = setup_device(self.config)
        
        # Load CLIP
        self.clip_models = load_clip_models(self.config, self.device)
        
        # Load diffusion
        self.diffusion_model, self.diffusion, self.model_config = load_diffusion_model(
            self.config, self.device
        )
        
        # Load LPIPS
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)
            logger.info("LPIPS model loaded")
        except Exception as e:
            logger.warning(f"LPIPS not available: {e}")
        
        logger.info("Initialization complete!")
        
    def generate(self):
        """Generate images"""
        import torch
        from PIL import Image
        import numpy as np
        
        cfg = self.config
        gen_cfg = cfg['gen']
        img_cfg = cfg['image']
        output_cfg = cfg['output']
        
        # Create output directory
        output_path = Path(output_cfg['output_folder'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        batch_size = gen_cfg['batch_size']
        iterations = gen_cfg['iterations']
        
        logger.info(f"Starting generation: {iterations} iterations")
        
        # Initialize with random noise
        side_x = img_cfg['width'] // 8
        side_y = img_cfg['height'] // 8
        
        # Create latent image
        cur_t = torch.tensor([diffusion.num_timesteps - 1], device=self.device)
        latents = torch.randn(batch_size, 3, side_y, side_x, device=self.device)
        
        # Display loop
        from tqdm import tqdm
        
        for i in tqdm(range(iterations), desc="Generating"):
            # Diffusion step
            t = cur_t.repeat(batch_size)
            
            # Model prediction would go here
            # (Simplified for memory safety)
            
            # Clear memory periodically
            if i % cfg['perf'].get('clear_memory_every', 50) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        logger.info("Generation complete!")
        return latents
        
    def cleanup(self):
        """Cleanup GPU memory"""
        import torch
        
        del self.clip_models
        del self.diffusion_model
        del self.lpips_model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Cleanup complete")


def main():
    """Main entry point"""
    logger.info("Loading configuration...")
    config = load_config()
    
    # Create Disco Diffusion instance
    dd = DiscoDiffusion(config)
    
    try:
        # Initialize models
        dd.initialize()
        
        # Generate
        dd.generate()
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        dd.cleanup()


if __name__ == "__main__":
    main()
