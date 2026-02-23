"""
Disco Diffusion - Optimized Version
AI Image Generation from Text Prompts

Optimizations:
- Standalone Python script (not Jupyter)
- Proper GPU memory management
- Configuration via config.py
- Error handling and recovery
- Progress tracking
"""

import os
import sys
import io
import json
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'CLIP'))
sys.path.append(str(PROJECT_ROOT / 'ResizeRight'))
sys.path.append(str(PROJECT_ROOT / 'guided-diffusion'))
sys.path.append(str(PROJECT_ROOT / 'SLIP'))


def check_dependencies():
    """Check if all required packages are installed"""
    # Map package names to import names
    package_map = {
        'torch': 'torch',
        'torchvision': 'torchvision', 
        'clip': 'clip',
        'lpips': 'lpips',
        'Pillow': 'PIL',
        'numpy': 'numpy',
        'tqdm': 'tqdm',
    }
    missing = []
    
    for pkg, imp in package_map.items():
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.info("Install with: pip install " + " ".join(missing))
        logger.info("Or: pip install torch torchvision pillow numpy tqdm lpips")
        return False
    return True


def setup_paths():
    """Setup and verify paths"""
    paths = {
        'root': PROJECT_ROOT,
        'output': PROJECT_ROOT / 'output',
        'init_images': PROJECT_ROOT / 'content' / 'init_images',
        'models': PROJECT_ROOT / 'content',
    }
    
    # Create directories if needed
    for name, path in paths.items():
        if name != 'root':
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
    
    return paths


def print_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     █████╗  ██████╗ ██████╗███████╗███████╗███╗   ███╗    ║
║    ██╔══██╗██╔════╝██╔════╝██╔════╝██╔════╝████╗ ████║    ║
║    ███████║██║     ██║     █████╗  ███████╗██╔████╔██║    ║
║    ██╔══██║██║     ██║     ██╔══╝  ╚════██║██║╚██╔╝██║    ║
║    ██║  ██║╚██████╗╚██████╗███████╗███████║██║ ╚═╝ ██║    ║
║    ╚═╝  ╚═╝ ╚═════╝ ╚═════╝╚══════╝╚══════╝╚═╝     ╚═╝    ║
║                                                              ║
║              AI Image Generation - Optimized                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main entry point"""
    print_banner()
    
    # Check dependencies
    logger.info("Checking dependencies...")
    if not check_dependencies():
        logger.error("Please install missing dependencies first!")
        sys.exit(1)
    
    # Setup paths
    logger.info("Setting up paths...")
    paths = setup_paths()
    
    # Print configuration
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("=" * 60)
    
    try:
        from config import (
            image_settings, text_prompts, model_settings,
            clip_settings, generation_settings, output_settings,
            performance_settings
        )
        
        logger.info(f"  Image Size: {image_settings['width']}x{image_settings['height']}")
        logger.info(f"  Iterations: {generation_settings['iterations']}")
        logger.info(f"  Batch Size: {generation_settings['batch_size']}")
        logger.info(f"  Device: {performance_settings['device']}")
        logger.info(f"  Output: {output_settings['output_folder']}")
        
        # Show prompts
        logger.info("\n  Text Prompts:")
        for prompt, weight in text_prompts.items():
            logger.info(f"    [{weight}] {prompt[:60]}...")
            
    except ImportError as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"\n✓ GPU: {gpu_name}")
            logger.info(f"  Total Memory: {gpu_mem:.1f} GB")
        else:
            logger.warning("\n⚠ No GPU detected, using CPU (will be slow)")
            performance_settings['device'] = 'cpu'
    except Exception as e:
        logger.warning(f"Could not check GPU: {e}")
    
    logger.info("=" * 60)
    
    # Ask to proceed
    logger.info("\nTo start generation, edit config.py with your prompts and settings,")
    logger.info("then run: python generate.py")
    logger.info("\nOr run generate.py directly with default settings.")
    
    # For now, just show that everything loaded correctly
    logger.info("\n✓ Configuration loaded successfully!")
    logger.info("\nNext step: Run 'python generate.py' to start generating images.")


if __name__ == "__main__":
    main()
