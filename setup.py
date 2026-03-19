"""
01_setup.py
-----------
Install dependencies and verify the environment.
Run this first in your Colab notebook cell:
  !python 01_setup.py
"""

import subprocess
import sys

PACKAGES = [
    "transformers>=4.40.0",
    "accelerate>=0.30.0",
    "bitsandbytes>=0.43.0",   # 4-bit / 8-bit quantization
    "datasets>=2.19.0",
    "torch>=2.2.0",
    "tqdm",
    "jsonlines",
]

def install():
    print("Installing packages...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", *PACKAGES
    ])
    print("Done.\n")

def verify():
    import torch
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gb = props.total_memory / 1e9
            print(f"  GPU {i}: {props.name}  {gb:.1f} GB VRAM")
    print()

    # Verify HF datasets
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train[:5]")
    print(f"GSM8K sample loaded OK — {len(ds)} rows")
    print(f"  Example: {ds[0]['question'][:80]}...")
    print()

    # Model recommendation based on VRAM
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print("Recommended model based on VRAM:")
        if vram_gb >= 70:
            print("  -> Qwen/QwQ-32B-Preview (full bf16, ~65GB) — best quality")
            print("     or deepseek-ai/DeepSeek-R1-Distill-Qwen-32B (same size)")
        elif vram_gb >= 35:
            print("  -> Qwen/QwQ-32B-Preview (4-bit, ~18GB) — good quality")
            print("     or deepseek-ai/DeepSeek-R1-Distill-Qwen-14B (bf16, ~28GB)")
        else:
            print("  -> deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (bf16, ~14GB)")
            print("     or Qwen/QwQ-32B-Preview (4-bit bitsandbytes)")

if __name__ == "__main__":
    install()
    verify()